function PUPIL_CLEAN = cleanPupil(PUPIL,TIME,TH)
%% function PUPIL_CLEAN = cleanPupil(PUPIL,TIME)
% % % INPUT
% PUPIL: structure containing 
%   X: x position of the pupil center in pixel
%   Y: y position of the pupil center in pixel
%   CONF: confidence on the measurement of the pupil, range 0 (bad) to 1 (good)
% TIME: timestamp vecor of the samples
% TH: threshold value
% % % OUTPUT
% PUPIL_CLEAN: structure containing the same fields, but filtered for noise

if nargin < 3
    TH = 0.7;
end

PUPIL_CLEAN = PUPIL;

s_num = length(PUPIL.X);

%% REMOVE SINGLE CONFIDENCE SPIKE
template = [-1 1 -1];
CONF_TMP = 2*PUPIL.CONF - 1;
CONF_TMP = conv(CONF_TMP,template,'same');

maskI = CONF_TMP < -1.5;
for i = 2:s_num-1
    if maskI(i)
        
        PUPIL_CLEAN.X(i) = 0.5*(PUPIL_CLEAN.X(i-1) + PUPIL_CLEAN.X(i+1));
        PUPIL_CLEAN.Y(i) = 0.5*(PUPIL_CLEAN.Y(i-1) + PUPIL_CLEAN.Y(i+1));
        
    end
end

%% CONFIDENCE THRESHOLD
PUPIL_CLEAN.CONF = smooth(smooth(hampel(PUPIL.CONF),5),9);

maskC = PUPIL_CLEAN.CONF < TH;

PUPIL_CLEAN.X(maskC) = nan;
PUPIL_CLEAN.Y(maskC) = nan;

