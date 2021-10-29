%% COMPUTE CLUSTER
function [CLUSTER_IDX,CENTROIDS,StartPoint] = computeCluster(X0,Y0,TIME,POINTS_NUM,RES,starting_time)


if nargin < 6
    starting_time = 0;
end

Rep = 5;

TIME(TIME<starting_time) = nan;

mask_nan = isnan(X0+Y0+TIME);
X0(mask_nan) = nan;
Y0(mask_nan) = nan;
TIME(mask_nan) = nan;
X = [X0,Y0,TIME];

if nargin < 3
    POINTS_NUM = 5;
    
    [dumb,StartPoint] = kmeans(X(1:5:end,:),POINTS_NUM,'Distance','cityblock',...
    'Replicates',Rep,'Options',statset('Display','off'));
end


%% SET START POINT FOR 9 POINTS CALIBRATION WITH CALIBRATION HELPER
low_v = 0.2;
mid_v = 0.5;
high_v = 0.8;
low_h = 0.15;
mid_h = 0.5;
high_h = 0.85;
StartPoint = [mid_h, mid_v;
              mid_h, high_v;
              low_h, high_v;
              low_h, mid_v;
              low_h, low_v;
              mid_h, low_v;
              high_h, low_v;
              high_h, mid_v;
              high_h, high_v;];
StartPoint(:,1) = StartPoint(:,1) .* RES(1);
StartPoint(:,2) = StartPoint(:,2) .* RES(2);


SHIFT = 1; 
MOTION = sqrt((X0 - circshift(X0,SHIFT)).^2 + (Y0 - circshift(Y0,SHIFT)).^2);
MOTION(1:SHIFT) = 0;MOTION(end-SHIFT:end) = 0;
POS = find(MOTION>(0.05*RES(2)));

[START_TIME, END_TIME] = find_periods(X0,Y0,TIME,POINTS_NUM,starting_time);


% % DTIME = diff(TIME);
% % POS = find(DTIME>3*nanmedian(DTIME));

% START_TIME = TIME([1; POS]); 
% END_TIME = [TIME(POS); nanmax(TIME)];
% StartTime = linspace(nanmin(TIME),nanmax(TIME),POINTS_NUM)';
StartTime = 0.5*(START_TIME + END_TIME);

% display(StartPoint)
% display(StartTime)
StartPoint = [StartPoint,StartTime];

%% ESTIMATE CLUSTERS
warning off
[CLUSTER_IDX,CENTROIDS] = kmeans(X,POINTS_NUM,'Distance','cityblock',...
    'Replicates',Rep,'Options',statset('Display','off'),'Start',repmat(StartPoint,[1 1 Rep]));




