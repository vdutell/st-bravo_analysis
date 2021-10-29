%% SELECT STEADY TARGETS
function IDX_SELECT = selectSteadyTargets(ELLIPSES,CENTROIDS,CLUSTER_IDX,MOTION_TH,WIN)

X = ELLIPSES.X0;
Y = ELLIPSES.Y0;

if nargin<5
    WIN = 1;
end

if nargin<4
    MOTION_TH = 10;
end

IDX_SELECT = [];

%% COMPUTE TARGET MOTION
XT = X; YT = Y;
MOTION = smooth(sqrt((XT - circshift(XT,-1)).^2 + (YT - circshift(YT,-1)).^2),5,'moving');


for  i = 1:size(CENTROIDS,1)

    XT = X;
    YT = Y;

    nan_mask = (MOTION > MOTION_TH) | ~(CLUSTER_IDX==i);
    
    XT(nan_mask) = nan;
    YT(nan_mask) = nan;
    
    % compute distance from centroid
    DIST = smooth(sqrt((XT - CENTROIDS(i,1)).^2 + (YT - CENTROIDS(i,2)).^2),'moving');

    % find minimum distance
    [M pos] = nanmin(DIST);
    
    % select (2*WIN+1) samples around the best
    IDX_SELECT = [IDX_SELECT ([-WIN:WIN] + pos)];
    
%     if VIEW
%         plot(XT(IDX_SELECT),YT(IDX_SELECT),'o','color',COLOURS(i,:))
%     end
end

IDX_SELECT = sort(IDX_SELECT);

