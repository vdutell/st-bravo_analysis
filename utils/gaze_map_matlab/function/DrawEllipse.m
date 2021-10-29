function [XE, YE, MASK] = DrawEllipse(x0,y0,a,b,theta,RES)

theta = -theta;
t = linspace(-pi , pi, 64);

% % x = x0 + a .* cos(t);
% % y = y0 + b .* sin(t);

x = a .* cos(t);
y = b .* sin(t);

R = [cosd(theta) -sind(theta); sind(theta) cosd(theta)];

XYr = [x; y]' * R;

xr = XYr(:,1);
yr = XYr(:,2);

XE = x0 + xr;
YE = y0 + yr;

if nargin < 6
    MASK = [];
else 
    [XX YY] = meshgrid((-RES(1)/2:RES(1)/2-1), (-RES(2)/2:RES(2)/2-1));
    
    XYr = [XX(:), YY(:)] * R;
    XXr = reshape(XYr(:,1),[RES(2) RES(1)]);
    YYr = reshape(XYr(:,2),[RES(2) RES(1)]);
    
    [XXt YYt] = meshgrid((1:RES(1)) - x0, (1:RES(2)) - y0);
    
    MASK1 = double((XXr.^2 ./ a.^2 + YYr.^2 ./ b.^2 ) < 1);
   
    MASK = (interp2(XX,YY,MASK1,XXt,YYt));
    MASK = MASK == 1;
end



% 
%  C0 = (cosd(theta)).^2 ./ a.^2 + (sind(theta)).^2 ./ b.^2;
%     C1 = (sind(theta)).^2 ./ a.^2 + (cosd(theta)).^2 ./ b.^2;
%     C2 = (sind(2*theta)) ./ a.^2 - (sind(2*theta)) ./ b.^2;
%     C3 = -2*x0 .* (cosd(theta)).^2 ./ a.^2 - y0 .* (sind(2*theta)) ./ a.^2 + ...
%          -2*x0 .* (sind(theta)).^2 ./ b.^2 + y0 .* (sind(2*theta)) ./ b.^2;
%     C4 = -x0 .* (sind(2*theta)) ./ a.^2 - 2*y0 .* (sind(theta)).^2 ./ a.^2 + ...
%          +x0 .* (sind(2*theta)) ./ b.^2 - 2*y0 .* (cosd(theta)).^2 ./ b.^2;
%     C5 = x0.^2 .* (cosd(theta)).^2 ./ a.^2 + ...
%          x0*y0 .* (sind(2*theta)) ./ a.^2 + ...
%          y0.^2 .* (sind(theta)).^2 ./ a.^2 + ...
%          x0.^2 .* (sind(theta)).^2 ./ b.^2 + ...
%         -x0*y0 .* (sind(2*theta)) ./ b.^2 + ...
%          y0.^2 .* (cosd(theta)).^2 ./ b.^2 -1;
%      
%      MASK = C0 .* XX.^2 + C1 .* YY.^2 + C2 .* XX.*YY + C3 .* XX + C4 .* YY > C5;  