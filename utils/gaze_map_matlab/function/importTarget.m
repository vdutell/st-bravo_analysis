function [ELLIPSE, TIME]= importTarget(file_name)

TARGET = importdata(file_name);

TIME.TARGET = TARGET(:,1);

ELLIPSE.X0 = TARGET(:,2);
ELLIPSE.Y0 = TARGET(:,3);
ELLIPSE.a = TARGET(:,4);
ELLIPSE.b = TARGET(:,5);
ELLIPSE.theta = TARGET(:,6);