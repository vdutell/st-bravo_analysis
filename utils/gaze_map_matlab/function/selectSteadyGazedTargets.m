%% SELECT STEADY TARGETS
function [IDX_SELECT PUPIL] = selectSteadyGazedTargets(ELLIPSES,CENTROIDS,PUPIL,CLUSTER_IDX,TARGET_MOTION_TH,WIN)

X = ELLIPSES.X0;
Y = ELLIPSES.Y0;

if nargin<6
    WIN = 1;
end

if nargin<5
    TARGET_MOTION_TH = 10;
end

IDX_SELECT = [];

%% COMPUTE TARGET MOTION
XT = X; YT = Y;
TARGET_MOTION = smooth(sqrt((XT - circshift(XT,-1)).^2 + (YT - circshift(YT,-1)).^2),2*WIN+1,'moving');

%% COMPUTE EYE MOTION
XE = PUPIL.TARGET_X;
YE = PUPIL.TARGET_Y;
EYE_MOTION = smooth(sqrt((XE - circshift(XE,-1)).^2 + (YE - circshift(YE,-1)).^2),2*WIN+1,'moving');

for  i = 1:size(CENTROIDS,1)

    XT = X;
    YT = Y;

    nan_mask = (TARGET_MOTION > TARGET_MOTION_TH) | ~(CLUSTER_IDX==i);
    
    XT(nan_mask) = nan;
    YT(nan_mask) = nan;
    
    % compute distance from centroid
    TARGET_DIST = smooth(sqrt((XT - CENTROIDS(i,1)).^2 + (YT - CENTROIDS(i,2)).^2),2*WIN+1,'moving');
    TARGET_DIST_NORM = TARGET_DIST ./ prctile(TARGET_DIST,95);
    
    
    EYE_DIST = EYE_MOTION;
    EYE_DIST(nan_mask) = nan;
    EYE_DIST_NORM = EYE_DIST ./ prctile(EYE_DIST,95);
    
    PUPIL_CONF = PUPIL.TARGET_CONF;
    PUPIL_CONF(nan_mask) = nan;
    PUPIL_CONF_NORM = PUPIL_CONF ./ prctile(PUPIL_CONF,95);
    
    % find minimum distance
    [M pos] = nanmin(TARGET_DIST_NORM.*EYE_DIST_NORM./PUPIL_CONF_NORM);
    
    % select (2*WIN+1) samples around the best
    IDX_SELECT_TARGET = [-WIN:WIN] + pos;
    
    IDX_SELECT_TARGET(IDX_SELECT_TARGET <= 0) = nan;
    IDX_SELECT_TARGET(IDX_SELECT_TARGET >= length(TARGET_MOTION)) = nan;
    
    PUPIL.TARGET_X_AVERAGE(i) = nanmedian(PUPIL.TARGET_X(IDX_SELECT_TARGET));
    PUPIL.TARGET_Y_AVERAGE(i) = nanmedian(PUPIL.TARGET_Y(IDX_SELECT_TARGET));
    
    IDX_SELECT = [IDX_SELECT IDX_SELECT_TARGET];
    
%     figure,plot([-WIN:WIN] + pos,XE([-WIN:WIN] + pos),'m')
%     if VIEW
%         plot(XT(IDX_SELECT),YT(IDX_SELECT),'o','color',COLOURS(i,:))
%     end
end

IDX_SELECT = sort(IDX_SELECT);

% IDX_SELECT(IDX_SELECT < 0) = [];
% IDX_SELECT(IDX_SELECT > ) = [];

PUPIL.TARGET_X_SELECT = PUPIL.TARGET_X(IDX_SELECT);
PUPIL.TARGET_Y_SELECT = PUPIL.TARGET_Y(IDX_SELECT);

