function [TRANSF_AZ, TRANSF_EL] = apply_pupil2gaze(TRANSF,PUPIL,type)

if nargin < 2
    type = 'poly22';
end

if strcmp(type,'affine')
    [TRANSF_AZ, TRANSF_EL] = transformPointsInverse(TRANSF.tform, PUPIL(:,1), PUPIL(:,2));
end


if strcmp(type,'poly22')
    TRANSF_AZ = TRANSF.AZ(PUPIL);
    TRANSF_EL = TRANSF.EL(PUPIL);
end
