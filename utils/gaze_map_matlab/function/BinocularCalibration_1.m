function CALIB = BinocularCalibration_1(LE,RE,CALIBRATION_POINTS_NUM,VIEW)
% function CALIB = BinocularCalibration(LE,RE,CALIBRATION_POINTS_NUM,VIEW)
% Compute 3D calibration, mapping pupil position to gaze direction
%  INPUT
% LE left eye data structure
% RE right eye data structure
% CALIBRATION_POINTS_NUM number of calibration target used (9)
% VIEW flag to turn on or of verbose viewing
%  OUTPUT
% CALIB structure containing calibration

POINT_X_TARGET = size(RE.TARGET_POS,1)/CALIBRATION_POINTS_NUM;

% %% Refine eye pos
% LE.POS(2) = 0.5*(LE.POS(2) + RE.POS(2));
% LE.POS(3) = 0.5*(LE.POS(3) + RE.POS(3));
% RE.POS(2) = LE.POS(2);
% RE.POS(3) = LE.POS(3);

% ZERO_GAZE_FIX = nanmedian([LE.TARGET_POS(1:3,:);RE.TARGET_POS(1:3,:)]);

%% CALIBRATE LEFT EYE
% translation to LE coordinates
CALIB.LE.TARGET_POINTS = LE.TARGET_POS - repmat(LE.POS,[size(LE.TARGET_POS,1),1]); %% world coordinates of target
CALIB.LE.TARGET_PUPIL = [LE.PUPIL.TARGET_X_SELECT,LE.PUPIL.TARGET_Y_SELECT]; %% pupil coordinates in image plane for eye looking at target

% Helmholtz rotation to central point (TARGET_POINTS(1,:)) (yaw, pitch, roll)
CALIB.LE.CENTRAL_POINT = nanmedian(CALIB.LE.TARGET_POINTS(1:POINT_X_TARGET,:));
% CENTRAL_POINT = ZERO_GAZE_FIX;

[CALIB.LE.R, CALIB.LE.Rinv] = computeXYZ_to_EULER(CALIB.LE.CENTRAL_POINT);

CALIB.LE.TARGET_POINTS_ROT = CALIB.LE.TARGET_POINTS*CALIB.LE.R;

% compute azimuth and elevation for calibration points
[~, ~, CALIB.LE.AZ, CALIB.LE.EL] = computeXYZ_to_EULER(CALIB.LE.TARGET_POINTS_ROT);

% fit transformation from PUPIL POSITION (x,y) to GAZE DIRECTION (AZ,EL)
CALIB.LE.pupil2gaze = estimate_pupil2gaze(CALIB.LE.TARGET_PUPIL,CALIB.LE.AZ,CALIB.LE.EL,'affine');

% compute rmse
[AZ_FIT, EL_FIT] = apply_pupil2gaze(CALIB.LE.pupil2gaze,CALIB.LE.TARGET_PUPIL,'affine');
% AZ_FIT = CALIB.LE.pupil2gaze.AZ(CALIB.LE.TARGET_PUPIL);
% EL_FIT = CALIB.LE.pupil2gaze.EL(CALIB.LE.TARGET_PUPIL);
CALIB.LE.RMSE = CALIB.LE.pupil2gaze.gof.RMSE; sqrt(nanmean((CALIB.LE.AZ-AZ_FIT).^2 + (CALIB.LE.EL-EL_FIT).^2));

CALIB.LE.POS = LE.POS;

%% CALIBRATE RIGHT EYE
% translation to LE coordinates
CALIB.RE.TARGET_POINTS = RE.TARGET_POS - repmat(RE.POS,[size(RE.TARGET_POS,1),1]);
CALIB.RE.TARGET_PUPIL = [RE.PUPIL.TARGET_X_SELECT,RE.PUPIL.TARGET_Y_SELECT];

% Helmholtz rotation to central point (TARGET_POINTS(1,:)) (yaw, pitch, roll)
CALIB.RE.CENTRAL_POINT = nanmedian(CALIB.RE.TARGET_POINTS(1:3,:));
% CENTRAL_POINT = ZERO_GAZE_FIX;

[CALIB.RE.R, CALIB.RE.Rinv] = computeXYZ_to_EULER(CALIB.RE.CENTRAL_POINT);

CALIB.RE.TARGET_POINTS_ROT = CALIB.RE.TARGET_POINTS*CALIB.RE.R;

% compute azimut and elevation for calibration points
[~, ~, CALIB.RE.AZ, CALIB.RE.EL] = computeXYZ_to_EULER(CALIB.RE.TARGET_POINTS_ROT);

% fit transformation from PUPIL POSITION (x,y) to GAZE DIRECTION (AZ,EL)
CALIB.RE.pupil2gaze = estimate_pupil2gaze(CALIB.RE.TARGET_PUPIL,CALIB.RE.AZ,CALIB.RE.EL,'affine');

% compute rmse
[AZ_FIT, EL_FIT] = apply_pupil2gaze(CALIB.RE.pupil2gaze,CALIB.RE.TARGET_PUPIL,'affine');
% AZ_FIT = CALIB.RE.pupil2gaze.AZ(CALIB.RE.TARGET_PUPIL);
% EL_FIT = CALIB.RE.pupil2gaze.EL(CALIB.RE.TARGET_PUPIL);
CALIB.RE.RMSE = sqrt(nanmean((CALIB.RE.AZ-AZ_FIT).^2 + (CALIB.RE.EL-EL_FIT).^2));

CALIB.RE.POS = RE.POS;






