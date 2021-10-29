function GAZE = computeMonoGaze(PUPIL,LE)
% GAZE = [AZIMUTH ELEVATION];

% AZIMUTH = LE.pupil2gaze.AZ([PUPIL.X PUPIL.Y]);
% ELEVATION = LE.pupil2gaze.EL([PUPIL.X PUPIL.Y]);
if isstruct(PUPIL)
    [TRANSF_AZ, TRANSF_EL] = apply_pupil2gaze(LE.pupil2gaze,[PUPIL.X PUPIL.Y],LE.pupil2gaze.type);
elseif isnumeric(PUPIL)
    [TRANSF_AZ, TRANSF_EL] = apply_pupil2gaze(LE.pupil2gaze,PUPIL,LE.pupil2gaze.type);
end

GAZE = [TRANSF_AZ TRANSF_EL];