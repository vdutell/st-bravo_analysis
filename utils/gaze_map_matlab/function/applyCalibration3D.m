function FixPoint3D = applyCalibration3D(ROOT,EXP_FOLDER,camera_parameters_file,VIEW)
% function FixPoint3D = computeFixPoint3D(ROOT,EXP_FOLDER,camera_parameters_file,VIEW)
%
%% INPUT
% ROOT root folder
% EXP_FOLDER experiment folder
% camera_parameters_file file containing camera parameters used
% VIEW flag to turn on or of verbose viewing
%% OUTPUT
% FixPoint3D 3D coordinate {x,y,z] for the binocular fixation point


addpath E:\POPOLAZIONE.M\FUNZIONI\NumPyPLUGIN\npy-matlab-master

if nargin < 4
    VIEW = false;
end

%% LOAD FILES
MIN_TIME = [];

% camera parameters
load([ROOT camera_parameters_file]);

% open video file
world_video = VideoReader([ROOT EXP_FOLDER '\world.mp4']);
FRAME_RES = [world_video.Width world_video.Height];

% load depth timestamps
TIME.DEPTH = importdata([ROOT EXP_FOLDER '\depth\timestamps.csv']);
MIN_TIME = [MIN_TIME min(TIME.DEPTH)];

% load pupil data
if ~exist([ROOT EXP_FOLDER '\pupil_1.csv'])
    system(['python ./utils/parseEYE_PL_Data.py ' ROOT EXP_FOLDER]);
end
RIGHT_EYE.DATA_RAW = importdata([ROOT EXP_FOLDER '\pupil_0.csv']);
RIGHT_EYE.PUPIL.X = RIGHT_EYE.DATA_RAW.data(:,3);
RIGHT_EYE.PUPIL.Y = RIGHT_EYE.DATA_RAW.data(:,4);

LEFT_EYE.DATA_RAW = importdata([ROOT EXP_FOLDER '\pupil_1.csv']);
LEFT_EYE.PUPIL.X = LEFT_EYE.DATA_RAW.data(:,3);
LEFT_EYE.PUPIL.Y = LEFT_EYE.DATA_RAW.data(:,4);

% TIME.LEFT_EYE = readNPY([ROOT EXP_FOLDER '\eye1_timestamps.npy']);
% TIME.RIGHT_EYE = readNPY([ROOT EXP_FOLDER '\eye0_timestamps.npy']);
TIME.LEFT_EYE = LEFT_EYE.DATA_RAW.data(:,2);
TIME.RIGHT_EYE = RIGHT_EYE.DATA_RAW.data(:,2);

MIN_TIME = [MIN_TIME min(TIME.LEFT_EYE) min(TIME.RIGHT_EYE)];


% load world camera
TIME.WORLD = readNPY([ROOT EXP_FOLDER '\world_timestamps.npy']);

MIN_TIME = [MIN_TIME min(TIME.WORLD)];


%% NORMALIZE AND CUT TIMESTAMPS AND REMOVE NANs
MIN_TIME = max(MIN_TIME);
MAX_TIME = min([max(TIME.WORLD) max(TIME.DEPTH) max(TIME.LEFT_EYE) max(TIME.RIGHT_EYE)]);
DURATION = MAX_TIME - MIN_TIME;

FN = fieldnames(TIME);
for n = 1:length(FN)
    eval(['TMP = TIME.' FN{n} ';'])
    TMP = TMP - MIN_TIME;
    TMP(TMP>DURATION) = NaN;
%     TMP(isnan(TMP)) = [];
    eval(['TIME_NORM.' FN{n} ' = TMP;'])
    
    if strcmp(FN{n},'WORLD')
        WORLD_POS = n;
    end
end

%% MATCH INDEXES TO DEPTH
IDX_2DEPTH = nan(length(TIME_NORM.DEPTH),length(FN));
TIME_TH = 1/10;
for n = 1:length(FN)
    for i = 1:length(TIME_NORM.DEPTH)
        
        eval(['[dumb pos] = min(abs(TIME_NORM.' FN{n} ' - TIME_NORM.DEPTH(i)));'])
        eval(['CHECK = (abs(TIME_NORM.' FN{n} '(pos) - TIME_NORM.DEPTH(i)) > TIME_TH) || isnan(dumb);'])

        if CHECK
            IDX_2DEPTH(i,n) = nan;
        else
            IDX_2DEPTH(i,n) = pos;
        end

    end
end

%% MATCH INDEXES LEFT TO RIGHT EYE
IDX_2L = nan(length(TIME_NORM.LEFT_EYE),length(FN));
TIME_TH = 1/10;
for n = 1:length(FN)
    for i = 1:length(TIME_NORM.LEFT_EYE)
        
        eval(['[dumb pos] = min(abs(TIME_NORM.' FN{n} ' - TIME_NORM.LEFT_EYE(i)));'])
        eval(['CHECK = (abs(TIME_NORM.' FN{n} '(pos) - TIME_NORM.LEFT_EYE(i)) > TIME_TH) || isnan(dumb);'])

        if CHECK
            IDX_2L(i,n) = nan;
        else
            IDX_2L(i,n) = pos;
        end

    end
end

% %% MATCH INDEXES LEFT TO RIGHT EYE
% IDX_2R = nan(length(TIME_NORM.RIGHT_EYE),length(FN));
% TIME_TH = 1/10;
% for n = 1:length(FN)
%     for i = 1:length(TIME_NORM.RIGHT_EYE)
%         
%         eval(['[dumb pos] = min(abs(TIME_NORM.' FN{n} ' - TIME_NORM.RIGHT_EYE(i)));'])
%         eval(['CHECK = (abs(TIME_NORM.' FN{n} '(pos) - TIME_NORM.RIGHT_EYE(i)) > TIME_TH) || isnan(dumb);'])
% 
%         if CHECK
%             IDX_2R(i,n) = nan;
%         else
%             IDX_2R(i,n) = pos;
%         end
% 
%     end
% end

%% APPLY MONOCULAR CALIBRATION
load([ROOT '\CalibrationFunction'])

LEFT_EYE.GAZE = computeMonoGaze(LEFT_EYE.PUPIL,CALIB.LE);
RIGHT_EYE.GAZE = computeMonoGaze(RIGHT_EYE.PUPIL,CALIB.RE);

% match right eye to left
for i = 1:length(IDX_2L)
    if isnan(IDX_2L(i,3))
        RIGHT_EYE.MATCHED_GAZE(i,:) = [NaN NaN];
    else
        RIGHT_EYE.MATCHED_GAZE(i,:) = RIGHT_EYE.GAZE(IDX_2L(i,3),:);
    end
end

BIN_EYE = computeFixPoint3D(LEFT_EYE.GAZE,RIGHT_EYE.MATCHED_GAZE,CALIB);

if VIEW
    depth_video = VideoWriter([ROOT EXP_FOLDER '\POST_PROCESSING\depth_video.avi']);
    open(depth_video);
end

%% REFINE FIXATION POINT BY PROJECTING IT INTO DEPTH
for i = 1:BIN_EYE.FixPointNUM
    
    display(['FRAME: ' num2str(i) ' / ' num2str(BIN_EYE.FixPointNUM)]);
    IDX_USED = IDX_2L(i,1);
    
    if ~isnan(IDX_USED)
        %% LOAD DEPTH FRAME
        depthFrame = double(readNPY([ROOT EXP_FOLDER 'depth\depth_frame_' num2str(IDX_USED) '.npy']));

        %% COMPUTE POINT CLOUD
        PC = computePointCloud(depthFrame,CAMERA,100,2000);
        
        %% COMPUTE FIXATION POINT DISTANCE TO POINT CLOUD
        X1 = BIN_EYE.FP(i,:); % POINT 1 on the line
        X2 = BIN_EYE.POS;     % POINT 2 on the line
        
        X = PC.X(:);
        Y = PC.Y(:);
        Z = PC.Z(:);
        
        X0 = [X Y Z];
        
        NUM = cross(X0 - repmat(X1,[size(X0,1),1]),X0 - repmat(X2,[size(X0,1),1]));
        NUM = sqrt(sum(NUM.^2,2));
        
        DEN = repmat(X2-X1,[size(X0,1),1]);
        DEN = sqrt(sum(DEN.^2,2));
        
        DIST = NUM ./ DEN;
        [Dmin pos] = nanmin(DIST);
          
%         figure,hold on
%         plot3(X,Z,Y,'.b')
%         plot3([BIN_EYE.POS(1) BIN_EYE.FP(i,1)],[BIN_EYE.POS(3) BIN_EYE.FP(i,3)],[BIN_EYE.POS(2) BIN_EYE.FP(i,2)],'k')
%         plot3(X(pos),Z(pos),Y(pos),'or')    
%         plot3(BIN_EYE.POS(1),BIN_EYE.POS(3),BIN_EYE.POS(2),'oc')
%         plot3(BIN_EYE.FP(i,1),BIN_EYE.FP(i,3),BIN_EYE.FP(i,2),'+c')
        
        [Ypos Xpos] = ind2sub(size(depthFrame),pos);
        
        if VIEW
            figure(1),clf,hold on
            imagesc(depthFrame)
            plot(Xpos,Ypos,'ow','markersize',10)
            plot(Xpos,Ypos,'+w','markersize',10)
            axis off ij

            frame = getframe(gcf);
            writeVideo(depth_video,frame);
        end

        FixPoint3D(:,i) = [PC.X(pos) PC.Y(pos) PC.Z(pos)];
    end
    
end

if VIEW
    close(depth_video)
end
