function testTime(ROOT,CALIB_FOLDER,EXP_FOLDER,camera_parameters_file,VIEW)
% function FixPoint3D = computeFixPoint3D(ROOT,CALIB_FOLDER,EXP_FOLDER,camera_parameters_file)
%
%% INPUT
% ROOT root folder
% CALIB_FOLDER calibration folder
% EXP_FOLDER experiment folder
% camera_parameters_file file containing camera parameters used
% VIEW flag to turn on or of verbose viewing
%% OUTPUT
% FixPoint3D 3D coordinate {x,y,z] for the binocular fixation point

addpath function\npy-matlab-master
addpath function

if nargin < 5
    VIEW = false;
end

%% LOAD FILES
% MIN_TIME = [];

% camera parameters
load([ROOT camera_parameters_file]);

% open video file
world_video = VideoReader([ROOT EXP_FOLDER '\world.mp4']);
FRAME_RES = [world_video.Width world_video.Height];

% load depth timestamps
TIME.DEPTH = importdata([ROOT EXP_FOLDER '\depth\timestamps.csv']);
% MIN_TIME = [MIN_TIME min(TIME.DEPTH)];

% load world camera
TIME.WORLD = readNPY([ROOT EXP_FOLDER '\world_timestamps.npy']);

% MIN_TIME = [MIN_TIME min(TIME.WORLD)];

%% ALIGN TIME
FN = fieldnames(TIME);
MIN_TIME = [];
for n = 1:length(FN)
    eval(['TMP = TIME.' FN{n} ';'])
    
    mask = diff(TMP) < 0;
    mask(end+1) = true;
    TMP(mask) = nan;
    
    TMP(1:10) = nan;
    
    MIN_TIME = [MIN_TIME min(TMP)];
end

MIN_TIME = max(MIN_TIME);
MAX_TIME = min([max(TIME.WORLD) max(TIME.DEPTH)]);
DURATION = MAX_TIME - MIN_TIME;

TIME_WORLD_OFFSET = MIN_TIME - min(TIME.WORLD); %% world camera starts TIME_WORLD_OFFSET before time zero

%% NORMALIZE AND CUT TIMESTAMPS AND REMOVE NANs
FN = fieldnames(TIME);
for n = 1:length(FN)
    eval(['TMP = TIME.' FN{n} ';'])
    TMP = TMP - MIN_TIME;
    TMP(TMP<0) = nan;
    
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


if VIEW
    
%     f = figure('visible', 'off');
    create_figure(1,FRAME_RES(1)/2,FRAME_RES(2),'p'),clf;
    
    AX_C = axes('position',[0 0.5 1.09 0.55]);hold on
    IM_C = imagesc(zeros(720,1280));
    axis off ij
    
    AX_D = axes('position',[0 0 1.09 0.55]);hold on
    IM_D = imagesc(zeros(720,1280));
    axis off ij
end

%% SHOW FIGURE
count = 0;
IDX_OFFSET = -10;
colorFrame = zeros(720,1280,3);
for i = 1:length(IDX_2DEPTH)
    
    
    display(['FRAME: ' num2str(i)]);
    TIME_DEPTH = TIME_NORM.DEPTH(i);
    
    
    %% DEPTH FRAME
    try
        depthFrame = double(readNPY([ROOT EXP_FOLDER '/depth/depth_frame_' num2str(i,'%08.f') '.npy']));
    catch
        aa = 0;
    end
    
    if VIEW
        
        %% COLOR IMAGE
        if ~isnan(TIME_DEPTH)
%             while (count < IDX_2DEPTH(i,WORLD_POS) + IDX_OFFSET)
%                 colorFrame = readFrame(world_video);
%                 count = count + 1;
%             end
            while(count <= IDX_2DEPTH(i,2))
                colorFrame = readFrame(world_video);
                count = count + 1;
            end
        else
            colorFrame = zeros(720,1280,3);
        end

        
        
        axis(AX_C)
        set(IM_C,'CData',colorFrame)
        axis equal off
        drawnow
        
        axes(AX_D);
        set(IM_D,'CData',depthFrame)
        axis equal off
        drawnow

        aa = 0;
    end
 
end

