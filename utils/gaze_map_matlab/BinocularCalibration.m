function RMSE = BinocularCalibration(ROOT,LE_FOLDER,RE_FOLDER,CALIBRATION_FOLDER,CALIBRATION_POINTS_NUM)
% function CALIB = BinocularCalibration(LE,RE,CALIBRATION_POINTS_NUM)
% Compute 3D calibration, mapping pupil position to gaze direction
%  INPUT
% LE_FOLDER left eye folder containing LE data structure
%   LE.TARGET_POS: 3D target position in scene camera coordinates
% RE_FOLDER right eye folder containing RE data structure
% CALIBRATION_FOLDER folder where to save the calibration data (usually
% equal to the folder for the left eye)
% CALIBRATION_POINTS_NUM number of calibration target used (9)
%  OUTPUT
% RMSE root mean squa structure containing calibration

%% LOAD
load([ROOT LE_FOLDER '\EYE_POSITION_LE.mat'])
load([ROOT RE_FOLDER '\EYE_POSITION_RE.mat'])

POINT_X_TARGET = size(RE.TARGET_POS,1)/CALIBRATION_POINTS_NUM;

% %% Refine eye pos
% LE.POS(2) = 0.5*(LE.POS(2) + RE.POS(2));
% LE.POS(3) = 0.5*(LE.POS(3) + RE.POS(3));
% RE.POS(2) = LE.POS(2);
% RE.POS(3) = LE.POS(3);


%% 2D PUPIL VECTOR
CALIB.LE.TARGET_PUPIL = [LE.PUPIL.TARGET_X_SELECT,LE.PUPIL.TARGET_Y_SELECT]; %% pupil coordinates in image plane for eye looking at target
CALIB.RE.TARGET_PUPIL = [RE.PUPIL.TARGET_X_SELECT,RE.PUPIL.TARGET_Y_SELECT];


% ZERO_GAZE_FIX = nanmedian([LE.TARGET_POS(1:3,:);RE.TARGET_POS(1:3,:)]);
%% CENTRAL POINT
CENTRAL_POINT = 0.5*(nanmedian(LE.TARGET_POS(1:POINT_X_TARGET,:)) + ...
                     nanmedian(RE.TARGET_POS(1:POINT_X_TARGET,:)));

%% TRANSLATION
% translation to LE coordinates
CALIB.LE.T = LE.POS;
CALIB.LE.TARGET_POINTS = LE.TARGET_POS - repmat(CALIB.LE.T,[size(LE.TARGET_POS,1),1]); %% world coordinates of target
% translation to RE coordinates
CALIB.RE.T = RE.POS;
CALIB.RE.TARGET_POINTS = RE.TARGET_POS - repmat(CALIB.RE.T,[size(RE.TARGET_POS,1),1]);

%% ROTATION
% Helmholtz rotation to central point (TARGET_POINTS(1,:)) (yaw, pitch, roll)
[CALIB.LE.R, CALIB.LE.T] = computeRT(CENTRAL_POINT,LE.POS);
[CALIB.RE.R, CALIB.RE.T] = computeRT(CENTRAL_POINT,RE.POS);

% CALIB.LE.CENTRAL_POINT.W = CENTRAL_POINT;
% CALIB.RE.CENTRAL_POINT.W = CENTRAL_POINT; 

% [CALIB.LE.R, CALIB.LE.Rinv] = computeXYZ_to_EULER(CALIB.LE.CENTRAL_POINT);
% [CALIB.RE.R, CALIB.RE.Rinv] = computeXYZ_to_EULER(CALIB.RE.CENTRAL_POINT);

% CALIB.LE.TARGET_POINTS_ROT = CALIB.LE.TARGET_POINTS*CALIB.LE.R;
% CALIB.RE.TARGET_POINTS_ROT = CALIB.RE.TARGET_POINTS*CALIB.RE.R;
CALIB.LE.TARGET_POINTS_ROT = RotTras(LE.TARGET_POS,CALIB.LE.R,CALIB.LE.T,'forth');
CALIB.RE.TARGET_POINTS_ROT = RotTras(RE.TARGET_POS,CALIB.RE.R,CALIB.RE.T,'forth');

CALIB.LE.TARGET_POINTS_ROT_DIST = sqrt(sum(CALIB.LE.TARGET_POINTS_ROT.^2,2));
CALIB.RE.TARGET_POINTS_ROT_DIST = sqrt(sum(CALIB.RE.TARGET_POINTS_ROT.^2,2));

%% MONOCULAR GAZE ANGLES
% compute azimuth and elevation for calibration points
[~, ~, CALIB.LE.AZ, CALIB.LE.EL] = computeXYZ_to_EULER(CALIB.LE.TARGET_POINTS_ROT);
[~, ~, CALIB.RE.AZ, CALIB.RE.EL] = computeXYZ_to_EULER(CALIB.RE.TARGET_POINTS_ROT);


%% PUPIL TO MONOCULAR GAZE TRANSFORMATION
% fit transformation from PUPIL POSITION (x,y) to GAZE DIRECTION (AZ,EL)
CALIB.LE.pupil2gaze = estimate_pupil2gaze(CALIB.LE.TARGET_PUPIL,CALIB.LE.AZ,CALIB.LE.EL,'affine');
CALIB.RE.pupil2gaze = estimate_pupil2gaze(CALIB.RE.TARGET_PUPIL,CALIB.RE.AZ,CALIB.RE.EL,'affine');

% compute rmse
[AZ_FIT, EL_FIT] = apply_pupil2gaze(CALIB.LE.pupil2gaze,CALIB.LE.TARGET_PUPIL,'affine');
CALIB.LE.RMSE = CALIB.LE.pupil2gaze.gof.RMSE; sqrt(nanmean((CALIB.LE.AZ-AZ_FIT).^2 + (CALIB.LE.EL-EL_FIT).^2));

% TARGET_FIT = computeEULER_to_XYZ(AZ_FIT, EL_FIT);
% TARGET_FIT - CALIB.LE.TARGET_POINTS_ROT ./ repmat(sqrt(sum((CALIB.LE.TARGET_POINTS_ROT.^2),2)),[1 3]);

[AZ_FIT, EL_FIT] = apply_pupil2gaze(CALIB.RE.pupil2gaze,CALIB.RE.TARGET_PUPIL,'affine');
CALIB.RE.RMSE = sqrt(nanmean((CALIB.RE.AZ-AZ_FIT).^2 + (CALIB.RE.EL-EL_FIT).^2));


CALIB.LE.POS = LE.POS;
CALIB.RE.POS = RE.POS;


RMSE = [CALIB.LE.RMSE CALIB.RE.RMSE];

%% CYCLOPEAN EYE
CALIB.CE.POS = 0.5 * (CALIB.LE.POS + CALIB.RE.POS);
[CALIB.CE.R, CALIB.CE.T] = computeRT(CENTRAL_POINT,CALIB.CE.POS);

%% SAVE
save_file_name = [ROOT LE_FOLDER '\BINOCULAR_CALIB'];
save(save_file_name,'CALIB')

