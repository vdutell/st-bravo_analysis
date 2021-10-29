function [START_TIME, END_TIME] = find_periods(X0,Y0,TIME,POINTS_NUM,starting_time)

if nargin < 5
    starting_time = 0;
end

nan_mask = isnan(X0 + Y0 + TIME);
X0(nan_mask) = [];
Y0(nan_mask) = [];
TIME(nan_mask) = [];


MIN_TIME_DIST = 0.5; %sec

FPS = 1 ./ nanmedian(diff(TIME));

SMOOTH_WINDOW = round(MIN_TIME_DIST .* FPS);

%% COMPUTE MOTION
SHIFT = 1;
MOTION = sqrt((X0 - circshift(X0,SHIFT)).^2 + (Y0 - circshift(Y0,SHIFT)).^2);
MOTION(1:SHIFT) = 0;MOTION(end-SHIFT:end) = 0;

TEST = diff(MOTION) .* diff(TIME);
TEST(TEST<0) = 0;

POS = find(TEST>nanstd(TEST));

START_TIME = TIME([1; POS]); 
END_TIME = [TIME(POS); nanmax(TIME)];
% MOTION = smooth(smooth(MOTION,SMOOTH_WINDOW),SMOOTH_WINDOW);


% [Hmotion,BINmotion] = hist(TEST,101);
% dTEST = diff(TEST);
% dTEST = dTEST - circshift(dTEST,-1);
% 
% dTEST = conv(dTEST,[0 0 -1 2 -1 0 0],'same');

% [~, upper_TH] = triangular_threshold(Hmotion,BINmotion);
% 
% POS = find(TEST>0.5*upper_TH);
% 
% figure,plot(smooth(smooth(MOTION,SMOOTH_WINDOW),SMOOTH_WINDOW))
% STEADY = 1./MOTION;

end

function [lower_TH, upper_TH, MAX, lower_TH_pos, upper_TH_pos, max_pos] = ...
    triangular_threshold(H,xi)
% function [lower_TH, upper_TH, MAX, lower_TH_pos, upper_TH_pos, max_pos] = ...
% triangular_threshold(H,xi)
% H: histogram
% xi: dominum
%
% lower_TH: lower threshold
% upper_TH: upper threshold
% MAX: histogram max
l = length(H);
Htmp = H;

Htmp(1:floor(l/3)) = 0;
Htmp(end-ceil(l/3):end) = 0;

[M idx] = max(Htmp);
MAX=xi(idx);
max_pos=idx;

win = 3;
xi_orig = xi;

% Compute lines distance
xi = [(1-max_pos):(length(H)-max_pos)]';

%% LOWER
m = (H(idx)-H(1))./(xi(idx)-xi(1));
q = H(1)-m.*xi(1);

Dmin = -(H - xi'.*m - q) ./ sqrt(1+m.^2); Dmin=smooth(Dmin,win-2);
Dmin(max_pos:end) = nan;

[L idxl] = max(Dmin);
lower_TH = xi_orig(idxl);
lower_TH_pos = idxl;


%% UPPER
m = (H(end)-H(idx))./(xi(end)-xi(idx));
q = H(end)-m.*xi(end);

Dmax = (H - xi'.*m - q) ./ sqrt(1+m.^2); Dmax=smooth(Dmax,win-2);
Dmax(1:max_pos)=nan;

[U idxu] = min(Dmax);
upper_TH = xi_orig(idxu);
upper_TH_pos = idxu;


end
