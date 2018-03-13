function SignDetection()
% Create system objects used for reading video, loading the video file, detecting signs, and displaying the results.
videoFile = 'sign.mp4';
load('stopSignsAndCars.mat')
stopSigns = stopSignsAndCars(:,1:2);
stopSigns.imageFilename = fullfile(toolboxdir('vision'),...
    'visiondata',stopSignsAndCars.imageFilename);
detector = trainACFObjectDetector(stopSigns,'NumStages',4);

obj = setupSystemObjects(videoFile);

% Create an empty array of tracks.
tracks = initializeTracks(); 

% ID of the next track.
nextId = 1; 

% Get dimensions of current video to calculate processing area
% Create a video reader
videoInfo = VideoReader(videoFile);
info = get(videoInfo);
vidWidth = info.Width;
vidHeight = info.Height;

% Set the global parameters.
option.ROI                  = [1 1 vidWidth vidHeight];  % A rectangle [x, y, w, h] that limits the processing area to ground locations.
option.scThresh             = 0.0;                       % A threshold to control the tolerance of error in estimating the scale of a detected signs. 
option.gatingThresh         = 0.9;                       % A threshold to reject a candidate match between a detection and a track.
option.gatingCost           = 100;                       % A large value for the assignment cost matrix that enforces the rejection of a candidate match.
option.costOfNonAssignment  = 10;                        % A tuning parameter to control the likelihood of creation of a new track.
option.timeWindowSize       = 16;                        % A tuning parameter to specify the number of frames required to stabilize the confidence score of a track.
option.confidenceThresh     = 2;                         % A threshold to determine if a track is true positive or false alarm.
option.ageThresh            = 8;                         % A threshold to determine the minimum length required for a track being true positive.
option.visThresh            = 0.6;                       % A threshold to determine the minimum visibility value for a track being true positive.

%Sets limit on how long to run the video.
stopFrame = 1629; 
for fNum = 1:stopFrame
    frame   = readFrame();
 
    [centroids, bboxes, scores] = detectSigns();
    
    predictNewLocationsOfTracks();    
    
    [assignments, unassignedTracks, unassignedDetections] = ...
        detectionToTrackAssignment();
    
    updateAssignedTracks();    
    updateUnassignedTracks();    
    deleteLostTracks();    
    createNewTracks();
    
    displayTrackingResults();

    % Exit the loop if the video player figure is closed.
    if ~isOpen(obj.videoPlayer)
        break;
    end
end




%% Create System Objects for the tracking

    function obj = setupSystemObjects(videoFile)
        % Initializes Video I/O
        % Create objects for reading a video from a file, drawing the 
        % detected and tracked people in each frame, and playing the video.
        
        % Create a video file reader.
        obj.reader = vision.VideoFileReader(videoFile, 'VideoOutputDataType', 'uint8');
        
        % Create a video player.
        obj.videoPlayer = vision.VideoPlayer('Position', [1, 1, 1280, 960]);                
        
   
    end


%% Initialize Tracks
% This function creates an array of tracks.  Each track is a structure
% to represent a moving object in the video.  A track is represented by a 
% struct with the following fields: 
% id: An integer id for the track
% color: the color of the track
% bboxes: A N x 4 matrix to represent the bounding box.
% scores: A N x 1 vector to record the classification score from the
% detector
% kalmanFilter: Kalman filter object used for motion tracking.
% age:  Number of frames since initialization.
% totalVisibleCountt: total number of frames  the object is visible.
% confidence:  A pair of numbers to represent how confident we are in
% the track.
% predPosition: Predicted position of bounding box in the next frame.

    function tracks = initializeTracks()
        % Create an empty array of tracks
        tracks = struct(...
            'id', {}, ...
            'color', {}, ...
            'bboxes', {}, ...
            'scores', {}, ...
            'kalmanFilter', {}, ...
            'age', {}, ...
            'totalVisibleCount', {}, ...
            'confidence', {}, ...            
            'predPosition', {});
    end
%% Read Video
% Read the next video frame from the video file.
    function frame = readFrame()
        frame = step(obj.reader);
    end

%% Detect Signs
% This function returns the centroids, bounding boxes and classifcation
% scores of the detected signs.  It will do this by performing filtering
% and non-maximum suppression of the output from the detector.  

% Field:
% centroids: A N x 2 matrix.
% bboxes: A N x 4 matrix
% scores: A N x 1 matrix which holds the classication scores for the
% corresponding frames.


    function [centroids, bboxes, scores] = detectSigns()
        % Resize the image to increase the resolution of the sign.
        % This helps in detecting signs that are further away.
        resizeRatio = 1;
        frame = imresize(frame, resizeRatio, 'Antialiasing',false);
        
        % Run ACF people detector within a region of interest to produce
        % detection candidates.
        [bboxes, scores] = detect(detector, frame, option.ROI, ...
            'WindowStride', 2,...
            'NumScaleLevels', 4, ...
            'SelectStrongest', false);
        
        % Apply non-maximum suppression.
        [bboxes, scores] = selectStrongestBbox(bboxes, scores, ...
                            'RatioType', 'Min', 'OverlapThreshold', 0.6);                               
        
        % Compute the centroids
        if isempty(bboxes)
            centroids = [];
        else
            centroids = [(bboxes(:, 1) + bboxes(:, 3) / 2), ...
                (bboxes(:, 2) + bboxes(:, 4) / 2)];
        end
    end

%% Predict New Locations of Existing Tracks
% This function uses a Kalman filter to predict where the centroid
% of each track will be in the current frame and will update the bounding
% box accordingly. 

    function predictNewLocationsOfTracks()
        for i = 1:length(tracks)
            % Get the last bounding box on this track
            % to help with prediction. 
            bbox = tracks(i).bboxes(end, :);
            
            % Predict the current location of the track.
            predictedCentroid = predict(tracks(i).kalmanFilter);
            
            % Shift the bounding box so that its center is at the predicted location.
            tracks(i).predPosition = [predictedCentroid - bbox(3:4)/2, bbox(3:4)];
        end
    end

%% Assign Detections to Tracks
% This function assigns object detections in the current frames to
% the existing frames.  This is done by minimizing cost.  Cost is computed
% using an overlap ration between the predicted bounding box and the
% detected bounding box.

    function [assignments, unassignedTracks, unassignedDetections] = ...
            detectionToTrackAssignment()
        
        % Compute the overlap ratio.
        % Compute the cost of assigning each detection
        % to each track.
        predBboxes = reshape([tracks(:).predPosition], 4, [])';
        cost = 1 - bboxOverlapRatio(predBboxes, bboxes);

        % Force the optimization step to ignore some matches by
        % setting the associated cost to be a large number. 
        cost(cost > option.gatingThresh) = 1 + option.gatingCost;

        % Solve the assignment problem.
        [assignments, unassignedTracks, unassignedDetections] = ...
            assignDetectionsToTracks(cost, option.costOfNonAssignment);
    end

%% Update Assigned Tracks
% This function updates the assigned track with the corresponding
% direction.
% It calls the correct method of the vision toolbox to correct the location
% guess.  It will then store ths new bounding box by taking the average
% size of recent boxes and increases the age and total visible count by
% one.  Finally, this function will adjust the convidence score.


    function updateAssignedTracks()
        numAssignedTracks = size(assignments, 1);
        for i = 1:numAssignedTracks
            trackIdx = assignments(i, 1);
            detectionIdx = assignments(i, 2);

            centroid = centroids(detectionIdx, :);
            bbox = bboxes(detectionIdx, :);
            
            % Correct the estimate of the object's location
            % using the new detection.
            correct(tracks(trackIdx).kalmanFilter, centroid);
            %Stablizies bounding by with averages of recent bounding boxes. 
            T = min(size(tracks(trackIdx).bboxes,1), 4);
            w = mean([tracks(trackIdx).bboxes(end-T+1:end, 3); bbox(3)]);
            h = mean([tracks(trackIdx).bboxes(end-T+1:end, 4); bbox(4)]);
            tracks(trackIdx).bboxes(end+1, :) = [centroid - [w, h]/2, w, h];
            
            % Update  age.
            tracks(trackIdx).age = tracks(trackIdx).age + 1;
            
            % Update  score history.
            tracks(trackIdx).scores = [tracks(trackIdx).scores; scores(detectionIdx)];
            
            % Update visible count.
            tracks(trackIdx).totalVisibleCount = ...
                tracks(trackIdx).totalVisibleCount + 1;
            
            % Adjust track confidence score.
            T = min(option.timeWindowSize, length(tracks(trackIdx).scores));
            score = tracks(trackIdx).scores(end-T+1:end);
            tracks(trackIdx).confidence = [max(score), mean(score)];
        end
    end

%% Update Unassigned Tracks
% This function makes each unassigned track invisible and increases its age
% by 1.  It also appeneds the predicted bounding box to the track and the
% confidence level will be set to 0.

    function updateUnassignedTracks()
        for i = 1:length(unassignedTracks)
            idx = unassignedTracks(i);
            tracks(idx).age = tracks(idx).age + 1;
            tracks(idx).bboxes = [tracks(idx).bboxes; tracks(idx).predPosition];
            tracks(idx).scores = [tracks(idx).scores; 0];
            
            T = min(option.timeWindowSize, length(tracks(idx).scores));
            score = tracks(idx).scores(end-T+1:end);
            tracks(idx).confidence = [max(score), mean(score)];
            
        end
    end

%% Delete Lost Tracks
% This function deletes tracks that have been invisible for too many frames
% in a row or ones that have been invisible for a large number of frames
% overall.

    function deleteLostTracks()
        if isempty(tracks)
            return;
        end        
        
        % Compute the fraction of the track's age for which it was visible.
        ages = [tracks(:).age]';
        totalVisibleCounts = [tracks(:).totalVisibleCount]';
        visibility = totalVisibleCounts ./ ages;
        
        % Check the maximum detection confidence score.
        confidence = reshape([tracks(:).confidence], 2, [])';
        maxConfidence = confidence(:, 1);

        % Find the indices of lost tracks.
        lostInds = (ages <= option.ageThresh & visibility <= option.visThresh) | ...
             (maxConfidence <= option.confidenceThresh);

        % Delete lost tracks.
        tracks = tracks(~lostInds);
    end

%% Create New Tracks
% This function simply creates a new track.
    function createNewTracks()
        unassignedCentroids = centroids(unassignedDetections, :);
        unassignedBboxes = bboxes(unassignedDetections, :);
        unassignedScores = scores(unassignedDetections);
        
        for i = 1:size(unassignedBboxes, 1)            
            centroid = unassignedCentroids(i,:);
            bbox = unassignedBboxes(i, :);
            score = unassignedScores(i);
            
            % Create a Kalman filter object.
            kalmanFilter = configureKalmanFilter('ConstantVelocity', ...
                centroid, [2, 1], [5, 5], 100);
            
            % Create a new track.
            newTrack = struct(...
                'id', nextId, ...
                'color', [255, 255, 255], ...
                'bboxes', bbox, ...
                'scores', score, ...
                'kalmanFilter', kalmanFilter, ...
                'age', 1, ...
                'totalVisibleCount', 1, ...
                'confidence', [score, score], ...
                'predPosition', bbox);
            
            % Add it to the array of tracks.
            tracks(end + 1) = newTrack; %#ok<AGROW>
            
            % Increment the next id.
            nextId = nextId + 1;
        end
    end

%% Display Tracking Results
%  This function draws a colore dbounding box for every visible track
% on the video frame.  The transparency of the box is based off of the
% confidence level.
    function displayTrackingResults()

        displayRatio = 4/3;
        frame = imresize(frame, displayRatio);
        
        if ~isempty(tracks),
            ages = [tracks(:).age]';        
            confidence = reshape([tracks(:).confidence], 2, [])';
            maxConfidence = confidence(:, 1);
            
			noDispInds = (ages < option.ageThresh & maxConfidence < option.confidenceThresh) | ...
                       (ages < option.ageThresh / 2);
                   
            for i = 1:length(tracks)
                if ~noDispInds(i)
                    
                    % scale bounding boxes for display
                    bb = tracks(i).bboxes(end, :);
                    bb(:,1:2) = (bb(:,1:2)-1)*displayRatio + 1;
                    bb(:,3:4) = bb(:,3:4) * displayRatio;
                    
					frame = insertObjectAnnotation(frame, ...
                                            'rectangle', bb, ...
											'Sign', ...
                                            'Color', [255, 255, 255], ...
											'LineWidth', 3);
                end
            end
        end
        
             
        step(obj.videoPlayer, frame);
        
    end

%%

end
