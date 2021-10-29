%% TEST COMPUTE EYE POSITION FOR BOTH EYES

clear
clc

addpath function
addpath E:\POPOLAZIONE.M\FUNZIONI\VARI
addpath E:\POPOLAZIONE.M\FUNZIONI\NumPyPLUGIN\npy-matlab-master

VIEW = true; % verbose viewing

% ROOT = 'G:/PUPIL_CALIBRATION_DATA/calibration_test_2_noximea/2021_03_22/TEST1/';
% LEFT_FOLDER = '001'; RIGHT_FOLDER = '004'; EXP_FOLDER = '005';

ROOT = 'G:\PUPIL_CALIBRATION_DATA\calibration_test_2_noximea\2021_03_22\TEST2\';
LEFT_FOLDER = '008'; 
RIGHT_FOLDER = '009'; 
EXP_FOLDER = '010';
CALIB_FOLDER = LEFT_FOLDER;

% ROOT = 'G:\PUPIL_CALIBRATION_DATA\calibration_test_2_noximea\2021_04_27\';
% LEFT_FOLDER = '008'; RIGHT_FOLDER = '009'; 

camera_parameters_file = 'CameraIntrinsic_D435I_720p_843112073163';
% camera_parameters_file = 'CameraIntrinsic_D435I_540p_937622071937';

CALIBRATION_POINTS_NUM = 9;


%% COMPUTE EYE POSITION
% LEFT_EYE = ComputeEyePos(ROOT, 'L', camera_parameters_file, CALIBRATION_POINTS_NUM, VIEW);
% RIGHT_EYE = ComputeEyePos(ROOT, 'R', camera_parameters_file, CALIBRATION_POINTS_NUM, VIEW);
% starting_time_LE = 30;
LEFT_EYE_POS  = ComputeEyePos3D(ROOT, LEFT_FOLDER, 'L', camera_parameters_file, CALIBRATION_POINTS_NUM, VIEW);
RIGHT_EYE_POS = ComputeEyePos3D(ROOT, RIGHT_FOLDER, 'R', camera_parameters_file, CALIBRATION_POINTS_NUM, VIEW);
BASELINE = sqrt(sum((LEFT_EYE_POS - RIGHT_EYE_POS).^2));

display(['Subject IPD: ' num2str(BASELINE)]);


%% COMPUTE GAZE CALIBRATION
CALIBRATION_RMSE = BinocularCalibration(ROOT,LEFT_FOLDER,RIGHT_FOLDER,CALIB_FOLDER,CALIBRATION_POINTS_NUM);
display(['BINOCULAR CALIBRATION: RMS Left Eye: ' num2str(CALIBRATION_RMSE(1)) 'deg - RMS Right Eye: ' num2str(CALIBRATION_RMSE(2)) ' deg'])



%% TEST BINOCULAR CALIBRATION
CREATE_VIDEO = true;
VIEW = false;
try
    FixPoint3D = Compute3DGaze(ROOT,CALIB_FOLDER,EXP_FOLDER,camera_parameters_file,CREATE_VIDEO,VIEW);
end

% try
%     FixPoint3D = Compute3DGaze(ROOT,CALIB_FOLDER,LEFT_FOLDER,camera_parameters_file,CREATE_VIDEO);
% end
% 
% try
%     FixPoint3D = Compute3DGaze(ROOT,CALIB_FOLDER,RIGHT_FOLDER,camera_parameters_file,CREATE_VIDEO);
% end
% 
% testTime(ROOT,CALIB_FOLDER,RIGHT_FOLDER,camera_parameters_file,CREATE_VIDEO);
