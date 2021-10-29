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

H = H(:);
xi = xi(:);

l = length(H);
Htmp = H;

% Htmp(1:floor(l/3)) = 0;
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

Dmin = -(H - xi.*m - q) ./ sqrt(1+m.^2); Dmin=smooth(Dmin,win-2);
Dmin(max_pos:end) = nan;

[L idxl] = max(Dmin);
lower_TH = xi_orig(idxl);
lower_TH_pos = idxl;


%% UPPER
m = (H(end)-H(idx))./(xi(end)-xi(idx));
q = H(end)-m.*xi(end);

Dmax = (H - xi.*m - q) ./ sqrt(1+m.^2); Dmax=smooth(Dmax,win-2);
Dmax(1:max_pos)=nan;

[U idxu] = min(Dmax);
upper_TH = xi_orig(idxu);
upper_TH_pos = idxu;


