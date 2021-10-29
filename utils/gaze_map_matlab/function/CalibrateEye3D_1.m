%% COMPUTE EYE POSITION
function EYE = CalibrateEye3D_1(ROOT,EYE_FOLDER,EYE_TAG,camera_parameters_file,CALIBRATION_POINTS_NUM,VIEW, starting_time)
%%
%% INPUT
% ROOT root folder
% EYE_TAG flag for left or right eye
% camera_parameters_file file containing camera parameters used
% CALIBRATION_POINTS_NUM number of calibration target used (9)
% VIEW flag to turn on or of verbose viewing
% starting_time time at which the calibration started 
%% OUTPUT
% EYE structure containing
% EYE.POS eye position
% EYE.TARGET_POS center of the targets used for calibration
% EYE.TARGET_NORMAL normal to the targets used for calibration
% EYE.TARGET_CLUSTER cluster index of the targets used for calibration
% EYE.CALIB_DATA data used to perform the calibration
% EYE.TRANSFORM function to transform the [x,y] coordinated of the pupil center into gaze angles

addpath E:\POPOLAZIONE.M\FUNZIONI\NumPyPLUGIN\npy-matlab-master

if nargin < 7
    starting_time = 0;
end

if nargin < 6
    VIEW = false;
end

if VIEW
    fig_segment = figure();
    fig_planes = figure();
    fig_point_cloud = figure();
end

TARGET_MOTION_TH = 10; % motion threshold (pixel per frame)
WIN = 5; %  2*number of samples per target

% % % 
% % % if strcmp(EYE_TAG,'L')
% % %     EYE_TAG = 'LE';
% % % elseif strcmp(EYE_TAG,'R')
% % %     EYE_TAG = 'RE';
% % % end

%% LOAD FILES
% camera parameters
load([ROOT camera_parameters_file]);

% open video file
world_video = VideoReader([ROOT EYE_FOLDER '/world.mp4']);
EYE.FRAME_RES = [world_video.Width world_video.Height];

% load target data
MIN_TIME = [];
TARGET = importdata([ROOT EYE_FOLDER '/3d_calibration/marker_center.csv']);
TIME.TARGET = TARGET(:,1);
MIN_TIME = [MIN_TIME min(TIME.TARGET)];
ELLIPSE.X0 = TARGET(:,2);
ELLIPSE.Y0 = TARGET(:,3);
ELLIPSE.a = TARGET(:,4);
ELLIPSE.b = TARGET(:,5);
ELLIPSE.theta = TARGET(:,6);

clear TARGET

% load depth timestamps
TIME.DEPTH = importdata([ROOT EYE_FOLDER '/depth/timestamps.csv']);
MIN_TIME = [MIN_TIME min(TIME.DEPTH)];

if ~exist([ROOT EYE_FOLDER '/offline_data/pupil_0.csv'])
    system('cd utils');
    folder_name = [ROOT EYE_FOLDER '/offline_data'];
    for i = 1:length(folder_name)
        if folder_name(i) == '\'
            folder_name(i) = '/';
        end
    end
    system(['python utils/parseEYE_PL_Data.py ' folder_name]);
    system('cd ..');
end
    
% load target data
if strcmp(EYE_TAG,'L')
    TIME.EYE = readNPY([ROOT EYE_FOLDER '/eye1_timestamps.npy']); 
    EYE.DATA_RAW = importdata([ROOT EYE_FOLDER '/offline_data/pupil_1.csv']);
else
    TIME.EYE = readNPY([ROOT EYE_FOLDER '/eye0_timestamps.npy']);
    EYE.DATA_RAW = importdata([ROOT EYE_FOLDER '/offline_data/pupil_0.csv']);
end
MIN_TIME = [MIN_TIME min(TIME.EYE)];

%% FILTER OUT BAD DATA
EYE.PUPIL.X = EYE.DATA_RAW.data(:,3);
EYE.PUPIL.Y = EYE.DATA_RAW.data(:,4);
EYE.PUPIL.CONF = EYE.DATA_RAW.data(:,8);

EYE.PUPIL = cleanPupil(EYE.PUPIL,TIME.EYE,0.2);

% load world camera
TIME.WORLD = readNPY([ROOT EYE_FOLDER '\world_timestamps.npy']);

MIN_TIME = [MIN_TIME min(TIME.WORLD)];

%% ALIGN TIME
MIN_TIME = max(MIN_TIME);
MAX_TIME = min([max(TIME.WORLD) max(TIME.TARGET) max(TIME.DEPTH) max(TIME.EYE)]);
DURATION = MAX_TIME - MIN_TIME;

TIME_WORLD_OFFSET = MIN_TIME - min(TIME.WORLD);

%% NORMALIZE AND CUT TIMESTAMPS AND REMOVE NANs
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

%
% figure,plot(TIME_NORM.WORLD,'.')
% hold on, plot(TIME_NORM.DEPTH,'.')
% hold on, plot(TIME_NORM.TARGET,'.')

%% MATCH INDEXES TO DEPTH
IDX_2DEPTH = nan(length(TIME_NORM.TARGET),length(FN));
TIME_TH = 1/10;
for n = 1:length(FN)
    for i = 1:length(TIME_NORM.TARGET)
        
        eval(['[dumb pos] = min(abs(TIME_NORM.' FN{n} ' - TIME_NORM.TARGET(i)));'])
        eval(['CHECK = (abs(TIME_NORM.' FN{n} '(pos) - TIME_NORM.TARGET(i)) > TIME_TH) || isnan(dumb);'])
        
        if CHECK
            IDX_2DEPTH(i,n) = nan;
        else
            IDX_2DEPTH(i,n) = pos;
        end

    end
end

% remove values of non-matching indexes
mask = isnan(sum(IDX_2DEPTH,2));
ELLIPSE.X0(mask) = nan;
ELLIPSE.Y0(mask) = nan;

%% CLUSTER TARGET POSITION
[CLUSTER_IDX,CENTROIDS,StartPoint] = computeCluster(ELLIPSE.X0,ELLIPSE.Y0,TIME_NORM.TARGET,...
    CALIBRATION_POINTS_NUM,[CAMERA.COLOR_SENSOR.width CAMERA.COLOR_SENSOR.height],starting_time);

%% GET AVAILABLE GAZE DATA
EYE.PUPIL.TARGET_X = nan(size(CLUSTER_IDX));
EYE.PUPIL.TARGET_Y = nan(size(CLUSTER_IDX));
EYE.PUPIL.TARGET_CONF = nan(size(CLUSTER_IDX));
for i = 1:CALIBRATION_POINTS_NUM
    EYE.PUPIL.TARGET_X(CLUSTER_IDX==i) = EYE.PUPIL.X(IDX_2DEPTH(CLUSTER_IDX==i,3));
    EYE.PUPIL.TARGET_Y(CLUSTER_IDX==i) = EYE.PUPIL.Y(IDX_2DEPTH(CLUSTER_IDX==i,3));
    EYE.PUPIL.TARGET_CONF(CLUSTER_IDX==i) = EYE.PUPIL.CONF(IDX_2DEPTH(CLUSTER_IDX==i,3)); 
end

% %% SELECT STEADY TARGETS
% DEPTH_IDX_ANALYZE = selectSteadyTargets(ELLIPSE,CENTROIDS,CLUSTER_IDX,MOTION_TH,WIN);
%% SELECT STEADY TARGETS AND GAZE
[DEPTH_IDX_ANALYZE EYE.PUPIL] = selectSteadyGazedTargets(ELLIPSE,CENTROIDS,EYE.PUPIL,CLUSTER_IDX,TARGET_MOTION_TH,WIN);


%% SHOW IMAGES
if VIEW
    
    COLOURS = colormap(lines(CALIBRATION_POINTS_NUM));
    XS = CAMERA.COLOR_SENSOR.width;
    YS = CAMERA.COLOR_SENSOR.height;
    
    figure();hold on,axis equal ij
    patch([1 1 XS XS 1],[1 YS YS 1 1],'w')
    xlim([1 XS]),ylim([1 YS])
    
    for  i = 1:CALIBRATION_POINTS_NUM
        h(i) = plot(ELLIPSE.X0(CLUSTER_IDX==i),ELLIPSE.Y0(CLUSTER_IDX==i),'.','color',COLOURS(i,:),'MarkerSize',12);
    end
    plot(CENTROIDS(:,1),CENTROIDS(:,2),'ko','MarkerSize',15)
    title(['Cluster Assignments and Centroids - ' EYE_FOLDER ' eye'])
    
    for nn = 1:length(StartPoint)
        plot(xlim,[1 1].*StartPoint(nn,2),'k:')
        plot([1 1].*StartPoint(nn,1),ylim,'k:')
    end
    
    if strcmp(EYE_TAG,'L')
        text(1000,50,'TOP/NASAL')
    else
        text(50,50,'TOP/NASAL')
    end
    
    figure(fig_planes);clf
    hold on,axis equal
    box on;
    set(gca,'xdir','reverse')
    %     set(gca,'zdir','reverse')
    
    plot3(0,0,0,'ko')
    quiver3(0,0,0,0,0,50,'k','linewidth',4)
    quiver3(0,0,0,0,50,0,'k','linewidth',4)
    quiver3(0,0,0,50,0,0,'k','linewidth',4)
    
    %% PUPIL POSITION IMAGE
    figure;hold on
    title(['Pupil center (pixels) - ' EYE_TAG ' eye'])
    
    for  i = 1:CALIBRATION_POINTS_NUM
        plot(EYE.PUPIL.TARGET_X_SELECT(1+[0:2*WIN] + (2*WIN+1)*(i-1)),EYE.PUPIL.TARGET_Y_SELECT(1+[0:2*WIN] + (2*WIN+1)*(i-1)),'.','color',COLOURS(i,:),'MarkerSize',12)
    end
    plot(EYE.PUPIL.TARGET_X_AVERAGE,EYE.PUPIL.TARGET_Y_AVERAGE,'ok','MarkerSize',15)
    
    set(gca,'xtick',0:24:192,'ytick',00:24:192)
    axis equal
    box on
    grid on
    
    xlim([0 192]),ylim([0 192])
    
    if strcmp(EYE_TAG,'L')
        text(50,10,'TOP/NASAL')
        axis ij
        set(gca,'xdir','reverse')
    else
        text(1,185,'TOP/NASAL')
    end
    
end

%% COMPUTE TARGET FITTING
count = 1;
for i = DEPTH_IDX_ANALYZE
    
    display(['Computing frame #' num2str(i)])
    
    %% COMPUTE ELLIPSE MASK
    [XE, YE, TARGET_MASK] = DrawEllipse(ELLIPSE.X0(i),ELLIPSE.Y0(i),ELLIPSE.a(i),ELLIPSE.b(i),ELLIPSE.theta(i),EYE.FRAME_RES);
    
    %% LOAD COLOR FRAME
    try
        while(world_video.CurrentTime < TIME_NORM.WORLD(IDX_2DEPTH(i,WORLD_POS)) + TIME_WORLD_OFFSET)
            colorFrame = readFrame(world_video);
        end
    catch
        aa=0;
    end
    
    %% LOAD DEPTH FRAME
    depthFrame = double(readNPY([ROOT EYE_FOLDER '\depth\depth_frame_' num2str(IDX_2DEPTH(i,2),'%08.f') '.npy']));
    
    %% COMPUTE POINT CLOUD
    PC = computePointCloud(depthFrame,CAMERA,100,2000);
    
    if VIEW
        create_figure(fig_point_cloud,18,18,'c');clf
        hold on
        plot3(PC.X(1:10:end),PC.Z(1:10:end),PC.Y(1:10:end),'.b')
        axis equal;    box on
        zlim([-600 600]);    xlim([-600 600]);    ylim([0 1650])
        plot3(PC.X(TARGET_MASK),PC.Z(TARGET_MASK),PC.Y(TARGET_MASK),'.m')
        baseline = 64;
        plot3(baseline/2, 0,-50, 'ok','markerfacecolor','g','markersize',6)
        plot3(-baseline/2, 0,-50, 'ok','markerfacecolor','r','markersize',6)
        plot3(0, 0, 0,'ok','markerfacecolor','k','markersize',6)
        
        view(-16,12)
        set(gca,'xdir','reverse')
    end
    
    %% COMPUTE PLANE FITTING
    [P(count,:) N(count,:), plane{count}] = fitPlane(PC,TARGET_MASK,[ELLIPSE.X0(i) ELLIPSE.Y0(i)]);
    TARGET_POS_2D(count,:) = [ELLIPSE.X0(i) ELLIPSE.Y0(i)];
    
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
        
        figure(fig_planes);hold on
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
% EYE.POS = computeOptimalIntersection(P,N);
EYE.POS = computeOptimalIntersectionRANSAC(P,N);

%% OUTPUT
EYE.TARGET_POS = P;
EYE.TARGET_POS_2D = TARGET_POS_2D;
EYE.TARGET_NORMAL = N;
EYE.TARGET_CLUSTER = CLUSTER_IDX;


%% SAVE WORKSPACE FOR CALIBRATION
save_file_name = [ROOT EYE_FOLDER '\EYE_POSITION_WS'];
var_string = [];
VAR_LIST = whos;

for  i = 1:length(VAR_LIST)
    if ~strcmp(VAR_LIST(i).class,'matlab.ui.Figure')
        var_string = [var_string ',''' VAR_LIST(i).name ''''];
    end
end

eval(['save(''' save_file_name '''' var_string ')'])

save_file_name = [ROOT EYE_FOLDER '\EYE_POSITION'];

%% SAVE CALIBRATION RESULT
TARGET_PUPIL_CENTER = [EYE.PUPIL.TARGET_X_AVERAGE',EYE.PUPIL.TARGET_Y_AVERAGE'];
EYE_POSITION = EYE.POS;
POINTS_NUM = 2*WIN+1;
for i = 1:CALIBRATION_POINTS_NUM
    TARGET_POSITION(i,:) = nanmedian(EYE.TARGET_POS(1:POINTS_NUM + (i-1).*POINTS_NUM,:));
end
save(save_file_name,'EYE_POSITION','TARGET_PUPIL_CENTER','TARGET_POSITION','EYE','DEPTH_IDX_ANALYZE');

%% SKETCH FIGURE
if VIEW
    sketchFig = figure();
    EYE_COLOR = [1 0 0];
    
    h = plotSketch(sketchFig,EYE.POS,EYE.TARGET_POS,EYE_COLOR);
    plotFrustum(CAMERA.COLOR_SENSOR.FRUSTUM,1.05.*nanmax(EYE.TARGET_POS(:,3)));
    
    
    
    figure(fig_planes);
    axis equal
    [Xeye,Yeye,Zeye] = sphere() ;
    surf(12*Xeye + EYE.POS(1),12*Zeye + EYE.POS(3),12*Yeye + EYE.POS(2),...
        'facecolor',[1 0 0],'facelighting','gouraud','edgecolor','none')
    
    plotFrustum(CAMERA.COLOR_SENSOR.FRUSTUM,1.05.*nanmax(EYE.TARGET_POS(:,3)));
    
    set(gca,'xdir','reverse')
    
end