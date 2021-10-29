%% COMPUTE EYE POSITION
function EYE = ComputeEyePos(ROOT,eye_flag,camera_parameters_file,CALIBRATION_POINTS_NUM,VIEW)
%%
%% INPUT
% ROOT root folder
% eye_flag flag for left or right eye
% camera_parameters_file file containing camera parameters used
% CALIBRATION_POINTS_NUM number of calibration target used (9)
% VIEW flag to turn on or of verbose viewing
%% OUTPUT
% EYE structure containing
% EYE.POS eye position
% EYE.TARGET_POS center of the targets used for calibration
% EYE.TARGET_NORMAL normal to the targets used for calibration
% EYE.TARGET_CLUSTER cluster index of the targets used for calibration

addpath E:\POPOLAZIONE.M\FUNZIONI\NumPyPLUGIN\npy-matlab-master

if nargin < 5
    VIEW = false;
end

if VIEW
    fig_segment = figure();
    fig_planes = figure();
end

MOTION_TH = 10; % motion threshold (pixel per frame)
WIN = 1; %  2*number of samples per target


if strcmp(eye_flag,'L')
    EYE_TAG = 'LE';
elseif strcmp(eye_flag,'R')
    EYE_TAG = 'RE';
end

%% LOAD FILES
% camera parameters
load([ROOT camera_parameters_file]);

% open video file
world_video = VideoReader([ROOT EYE_TAG '\world.mp4']);
EYE.FRAME_RES = [world_video.Width world_video.Height];

% load target data
MIN_TIME = [];
TARGET = importdata([ROOT EYE_TAG '\3d_calibration\marker_center.csv']);
TIME.TARGET = TARGET(:,1);
MIN_TIME = [MIN_TIME min(TIME.TARGET)];
ELLIPSE.X0 = TARGET(:,2);
ELLIPSE.Y0 = TARGET(:,3);
ELLIPSE.a = TARGET(:,4);
ELLIPSE.b = TARGET(:,5);
ELLIPSE.theta = TARGET(:,6);

clear TARGET

% load timestamp
% TIME.EYE0 = readNPY([ROOT RECORDING_NUM '\eye0_timestamps.npy']);
% TIME.EYE1 = readNPY([ROOT RECORDING_NUM '\eye1_timestamps.npy']);
TIME.DEPTH = importdata([ROOT EYE_TAG '\depth\timestamps.csv']);
MIN_TIME = [MIN_TIME min(TIME.DEPTH)];

TIME.WORLD = readNPY([ROOT EYE_TAG '\world_timestamps.npy']);
MIN_TIME = [MIN_TIME min(TIME.WORLD)];
MIN_TIME = max(MIN_TIME);
TIME_WORLD_OFFSET = MIN_TIME - min(TIME.WORLD);

MAX_TIME = min([max(TIME.WORLD) max(TIME.TARGET) max(TIME.DEPTH)]) - min(MIN_TIME);

%% NORMALIZE AND CUT TIMESTAMPS
FN = fieldnames(TIME);
for n = 1:length(FN)
    eval(['TMP = TIME.' FN{n} ';'])
    TMP = TMP -MIN_TIME;
    TMP(TMP>MAX_TIME) = NaN;
    eval(['TIME_NORM.' FN{n} ' = TMP;'])
end
% 
% figure,plot(TIME_NORM.WORLD,'.')
% hold on, plot(TIME_NORM.DEPTH,'.')
% hold on, plot(TIME_NORM.TARGET,'.')

%% MATCH INDEXES
IDX = nan(length(TIME_NORM.TARGET),length(FN));
TIME_TH = 1/10;
for n = 1:length(FN)
    for i = 1:length(TIME_NORM.TARGET)
        %     tmp = find((TIME_NORM.TARGET < TIME_NORM.DEPTH(i)) & (TIME_NORM.TARGET > TIME_NORM.DEPTH(i-1)), 1);

        
        eval(['[dumb pos] = min(abs(TIME_NORM.' FN{n} ' - TIME_NORM.TARGET(i)));'])
        eval(['CHECK = abs(TIME_NORM.' FN{n} '(pos) - TIME_NORM.TARGET(i)) > TIME_TH;'])
%         eval(['tmp = find((TIME_NORM.' FN{n} ' <= TIME_NORM.TARGET(i)) & (TIME_NORM.' FN{n} ' >= TIME_NORM.TARGET(i-1)), 1);'])
        if CHECK
            IDX(i,n) = nan;
        else
            IDX(i,n) = pos;
        end
    end
end

% remove values of non-matching indexes
mask = isnan(sum(IDX,2));
ELLIPSE.X0(mask) = nan;
ELLIPSE.Y0(mask) = nan;

%% CLUSTER TARGET POSITION
[CLUSTER_IDX,CENTROIDS,StartPoint] = computeCluster(ELLIPSE.X0,ELLIPSE.Y0,TIME_NORM.TARGET,...
         CALIBRATION_POINTS_NUM,[CAMERA.COLOR_SENSOR.width CAMERA.COLOR_SENSOR.height]);


%% SELECT STEADY TARGETS
IDX_ANALYZE = selectSteadyTargets(ELLIPSE,CENTROIDS,CLUSTER_IDX,MOTION_TH,WIN);

%% SHOW IMAGES
if VIEW
    COLOURS = colormap(lines(CALIBRATION_POINTS_NUM));
    XS = CAMERA.COLOR_SENSOR.width;
    YS = CAMERA.COLOR_SENSOR.height;
    
    figure(),hold on,axis equal off
    patch([1 1 XS XS 1],[1 YS YS 1 1],'w')
    xlim([1 XS]),ylim([1 YS])
    
    for  i = 1:CALIBRATION_POINTS_NUM
        h(i) = plot(ELLIPSE.X0(CLUSTER_IDX==i),ELLIPSE.Y0(CLUSTER_IDX==i),'.','color',COLOURS(i,:),'MarkerSize',12);
    end
    plot(CENTROIDS(:,1),CENTROIDS(:,2),'ko','MarkerSize',15)
    title 'Cluster Assignments and Centroids'
    
    for nn = 1:length(StartPoint)
        plot(xlim,[1 1].*StartPoint(nn,2),'k:')
        plot([1 1].*StartPoint(nn,1),ylim,'k:')
    end

    figure(fig_planes),hold on,axis equal
    set(gca,'zdir','reverse')
    plot3(0,0,0,'ko')
    quiver3(0,0,0,0,0,50,'k','linewidth',4)
    quiver3(0,0,0,0,50,0,'k','linewidth',4)
    quiver3(0,0,0,50,0,0,'k','linewidth',4)
end

%% COMPUTE TARGET FITTING
count = 1;
for i = IDX_ANALYZE
    
    display(['Computing frame #' num2str(i)])
    
    %% COMPUTE ELLIPSE MASK
    [XE, YE, TARGET_MASK] = DrawEllipse(ELLIPSE.X0(i),ELLIPSE.Y0(i),ELLIPSE.a(i),ELLIPSE.b(i),ELLIPSE.theta(i),    EYE.FRAME_RES);
    
    %% LOAD COLOR FRAME
    try
        while(world_video.CurrentTime < TIME_NORM.WORLD(IDX(i,3)) + TIME_WORLD_OFFSET)
            
            colorFrame = readFrame(world_video);
        end
    catch
        aa=0;
    end
    
    %% LOAD DEPTH FRAME
    depthFrame = double(readNPY([ROOT EYE_TAG '\depth\depth_frame_' num2str(IDX(i,2)) '.npy']));
    
    %% COMPUTE POINT CLOUD
    PC = computePointCloud(depthFrame,CAMERA,100,2000);
    
    %% COMPUTE PLANE FITTING
    [P(count,:) N(count,:), plane{count}] = fitPlane(PC,TARGET_MASK,[ELLIPSE.X0(i) ELLIPSE.Y0(i)]);
    
    if VIEW
        create_figure(fig_segment,16,18,'c');clf
        subplot(211),hold on
        image(colorFrame)
        plot(XE,YE,'r','linewidth',4)
        plot(ELLIPSE.X0(i),ELLIPSE.Y0(i),'+r','linewidth',2)
        axis equal ij off
        
        
        subplot(212),hold on
        imagesc(depthFrame)
        plot(XE,YE,'r','linewidth',4)
        plot(ELLIPSE.X0(i),ELLIPSE.Y0(i),'+r','linewidth',2)
        axis equal ij off
        caxis(prctile(double(depthFrame(:)),[10 90]))
        
        drawnow
        
        figure(fig_planes),hold on
        QUIVER_GAIN = -700;
        warning off
        [xTarget, yTarget, zTarget] = prepareSurfaceData(PC.X(TARGET_MASK),PC.Y(TARGET_MASK),PC.Z(TARGET_MASK));
        plot3(xTarget(1:10:end), zTarget(1:10:end), yTarget(1:10:end), '.')
        plot3(xTarget(1:10:end), plane{count}(xTarget(1:10:end),yTarget(1:10:end)),yTarget(1:10:end),'.')
        quiver3(P(count,1),P(count,3),P(count,2),QUIVER_GAIN.*N(count,1),QUIVER_GAIN.*N(count,3),QUIVER_GAIN.*N(count,2))
    end
    
    count = count + 1;
    
end


%% COMPUTE EYE POSITION
% [EYE.POS, INT_POINTS, WEIGHTS] = computeBestIntersection(P,N);
EYE.POS = computeOptimalIntersection(P,N);

%% OUTPUT
EYE.TARGET_POS = P;
EYE.TARGET_NORMAL = N;
EYE.TARGET_CLUSTER = CLUSTER_IDX;


%% SAVE CALIBRATION WORKSPACE
save_file_name = [ROOT EYE_TAG '\EYE_POSITION'];
var_string = [];
VAR_LIST = whos;

for  i = 1:length(VAR_LIST)
    if ~strcmp(VAR_LIST(i).class,'matlab.ui.Figure')
        var_string = [var_string ',''' VAR_LIST(i).name ''''];
    end
end

eval(['save(''' save_file_name '''' var_string ')'])

%% SKETCH FIGURE
if VIEW
    sketchFig = figure();
    EYE_COLOR = [1 0 0];
    
    h = plotSketch(sketchFig,EYE.POS,EYE.TARGET_POS,EYE_COLOR);
    plotFrustum(CAMERA.COLOR_SENSOR.FRUSTUM,1.05.*nanmax(EYE.TARGET_POS(:,3)));
    
    
    
    figure(fig_planes)
    set(gca,'zdir','reverse')
    axis equal
    [Xeye,Yeye,Zeye] = sphere() ;
    surf(12*Xeye + EYE.POS(1),12*Zeye + EYE.POS(3),12*Yeye + EYE.POS(2),...
        'facecolor',[1 0 0],'facelighting','gouraud','edgecolor','none')
    
    plotFrustum(CAMERA.COLOR_SENSOR.FRUSTUM,1.05.*nanmax(EYE.TARGET_POS(:,3)));
    
end