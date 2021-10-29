%% FIT PLANE TO POINT CLOUD
function [P, N, plane, gof] = fitPlane(PC,MASK,ANCHOR_POINT)
%% INPUT
% PC point cloud
% MASK mask for selected point
% ANCHOR_POINT point of the plaen (center of the target)
%% OUTPUT
% P plane principal point
% N normapl to plane
% plane fitted object
% gof goodness of fit

if nargin<3
    X0 = round(sqrt(numel(PC.X))/2);
    Y0 = round(sqrt(numel(PC.Y))/2);
else
    X0 = round(ANCHOR_POINT(1));
    Y0 = round(ANCHOR_POINT(2));
end
%% Prepare plane fitting
ft = fittype( '(-a * x - b * y + d) ./ c', 'independent', {'x', 'y'}, 'dependent', 'z' );
opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
opts.Algorithm = 'Levenberg-Marquardt';
opts.Display = 'Off';
opts.Robust = 'Bisquare';
opts.StartPoint = [0.528075461338699 0.147384577385648 0.621015939489747 0.621015939489747];

[xTarget, yTarget, zTarget] = prepareSurfaceData(PC.X(MASK),PC.Y(MASK),PC.Z(MASK));

%% PERFORM FITTING
[plane, gof] = fit( [xTarget(1:10:end), yTarget(1:10:end)], zTarget(1:10:end), ft, opts );

%% COMPUTE POINT AND NORMAL
P = [PC.X(Y0,X0),PC.Y(Y0,X0),PC.Z(Y0,X0)];
DEN = sqrt(plane.a.^2 + plane.b.^2 + plane.c.^2);
N = [plane.a plane.b plane.c] ./ DEN;
