function [R, T] = computeRT(C,P)

T = P;                      %% Translation vector
C = C - T;                  %% Translate target point
R = computeXYZ_to_EULER(C); %% Rotate to translated target point


