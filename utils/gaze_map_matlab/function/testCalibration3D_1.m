function FixPoint3D = testCalibration3D_1(ROOT,EXP_FOLDER,camera_parameters_file,VIEW)
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
if ~exist([ROOT EXP_FOLDER '/offline_data/pupil_1.csv'])
    system(['python ./utils/parseEYE_PL_Data.py ' ROOT EXP_FOLDER '/offline_data']);
end
RIGHT_EYE.DATA_RAW = importdata([ROOT EXP_FOLDER '/offline_data/pupil_0.csv']);
RIGHT_EYE.PUPIL.X = RIGHT_EYE.DATA_RAW.data(:,3);
RIGHT_EYE.PUPIL.Y = RIGHT_EYE.DATA_RAW.data(:,4);
RIGHT_EYE.PUPIL.CONF = RIGHT_EYE.DATA_RAW.data(:,8);

TIME.RIGHT_EYE = RIGHT_EYE.DATA_RAW.data(:,2);

RIGHT_EYE.PUPIL = cleanPupil(RIGHT_EYE.PUPIL,TIME.RIGHT_EYE,0.8);

LEFT_EYE.DATA_RAW = importdata([ROOT EXP_FOLDER '/offline_data/pupil_1.csv']);
LEFT_EYE.PUPIL.X = LEFT_EYE.DATA_RAW.data(:,3);
LEFT_EYE.PUPIL.Y = LEFT_EYE.DATA_RAW.data(:,4);
LEFT_EYE.PUPIL.CONF = LEFT_EYE.DATA_RAW.data(:,8);

TIME.LEFT_EYE = LEFT_EYE.DATA_RAW.data(:,2);

LEFT_EYE.PUPIL = cleanPupil(LEFT_EYE.PUPIL,TIME.RIGHT_EYE,0.8);



MIN_TIME = [MIN_TIME min(TIME.LEFT_EYE) min(TIME.RIGHT_EYE)];


% load world camera
TIME.WORLD = readNPY([ROOT EXP_FOLDER '\world_timestamps.npy']);

MIN_TIME = [MIN_TIME min(TIME.WORLD)];

%% ALIGN TIME
MIN_TIME = max(MIN_TIME);
MAX_TIME = min([max(TIME.WORLD) max(TIME.DEPTH) max(TIME.RIGHT_EYE) max(TIME.LEFT_EYE)]);
DURATION = MAX_TIME - MIN_TIME;

TIME_WORLD_OFFSET = MIN_TIME - min(TIME.WORLD); %% world camera starts TIME_WORLD_OFFSET before time zero

%% NORMALIZE AND CUT TIMESTAMPS AND REMOVE NANs
FN = fieldnames(TIME);
for n = 1:length(FN)
    eval(['TMP = TIME.' FN{n} ';'])
    TMP = TMP - MIN_TIME;
    %     TMP(TMP>DURATION) = NaN;
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

%% MATCH ALL INDEXES TO LEFT EYE
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



%% APPLY MONOCULAR CALIBRATION
load([ROOT '\CalibrationFunction'])

LEFT_EYE.GAZE = computeMonoGaze(LEFT_EYE.PUPIL,CALIB.LE);
RIGHT_EYE.GAZE = computeMonoGaze(RIGHT_EYE.PUPIL,CALIB.RE);

LEFT_EYE.GAZE = smoothGaze(LEFT_EYE.GAZE);
RIGHT_EYE.GAZE = smoothGaze(RIGHT_EYE.GAZE);

% get right eye matched to left
for i = 1:length(IDX_2L)
    if isnan(IDX_2L(i,2))
        RIGHT_EYE.MATCHED_GAZE(i,:) = [NaN NaN];
    else
        RIGHT_EYE.MATCHED_GAZE(i,:) = RIGHT_EYE.GAZE(IDX_2L(i,2),:);
    end
end


% for i =
BIN_EYE = computeFixPoint3D(LEFT_EYE.GAZE,RIGHT_EYE.MATCHED_GAZE,CALIB);

if VIEW
    if ~exist([ROOT EXP_FOLDER '\POST_PROCESSING'])
        mkdir([ROOT EXP_FOLDER '\POST_PROCESSING'])
    end
    depth_video = VideoWriter([ROOT EXP_FOLDER '\POST_PROCESSING\depth_video.avi']);
    depth_video.FrameRate = 1./nanmedian(diff(TIME.LEFT_EYE));
    open(depth_video);
    
    create_figure(1,16,16,'c'),clf
    AX_C = axes('position',[0 0.5 1.09 0.55]);hold on
    IM_C = imagesc(zeros(720,1280));
    CIRC_C = plot(100,100,'or','markersize',10,'markerfacecolor','c')
    CROSS_C = plot(100,100,'+r','markersize',10)
    axis off ij
    
    AX_D = axes('position',[0 0 1.09 0.55]);hold on
    IM_D = imagesc(zeros(720,1280));
    CIRC_D = plot(100,100,'or','markersize',10,'markerfacecolor','c')
    CROSS_D = plot(100,100,'+r','markersize',10)
    axis off ij
end

%% REFINE FIXATION POINT BY PROJECTING IT INTO DEPTH
count = 0;
IDX_OFFSET = 2;
for i = 1:BIN_EYE.FixPointNUM
    
    display(['FRAME: ' num2str(i) ' / ' num2str(BIN_EYE.FixPointNUM)]);
    DEPTH_IDX = IDX_2L(i,1); %% DEPTH IDX
    COLOR_IDX = IDX_2L(i,4); %% COLOR IDX
    TIME_DEPTH = TIME_NORM.DEPTH(DEPTH_IDX);
    
    if ~isnan(DEPTH_IDX)
        
        %% DEPTH FRAME
        depthFrame = double(readNPY([ROOT EXP_FOLDER '/depth/depth_frame_' num2str(DEPTH_IDX,'%08.f') '.npy']));
        
        %% COMPUTE POINT CLOUD
        PC = computePointCloud(depthFrame,CAMERA,100,2000);
        
        %% COMPUTE FIXATION POINT DISTANCE TO POINT CLOUD
        X1 = BIN_EYE.FP(i,:);   % POINT 1 of the line
        X2 = CALIB.CE.POS;      % POINT 2 of the line
        
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
            
            %% COLOR IMAGE
            try
                while (count < IDX_2DEPTH(DEPTH_IDX,WORLD_POS) + IDX_OFFSET)
                    colorFrame = readFrame(world_video);
                    count = count + 1;
                end
%                 while(world_video.CurrentTime < TIME_NORM.WORLD(IDX_2DEPTH(DEPTH_IDX,WORLD_POS)) - TIME_DEPTH + TIME_WORLD_OFFSET)
%                     colorFrame = readFrame(world_video);
%                     count = count + 1;
%                 end
            catch
                aa=0;
            end
            %        %% IDENTIFY TARGET CENTER
            %         accum = CircularHough_Grd(rgb2gray(colorFrame), [100 300]);
            %         [~, posTarget]=max2(accum);
            
            axes(AX_C);
            set(IM_C,'CData',colorFrame)
            set(CROSS_C,'XData',Xpos,'YData',Ypos)
            set(CIRC_C,'XData',Xpos,'YData',Ypos)
            axis equal off
            
            axes(AX_D);
            set(IM_D,'CData',depthFrame)
            set(CROSS_D,'XData',Xpos,'YData',Ypos)
            set(CIRC_D,'XData',Xpos,'YData',Ypos)
            axis equal off
            
            frame = getframe(gcf);
            writeVideo(depth_video,frame);
        end
        
        FixPoint3D(:,i) = [PC.X(pos) PC.Y(pos) PC.Z(pos)];
    end
    
end

if VIEW
    close(depth_video)
end
