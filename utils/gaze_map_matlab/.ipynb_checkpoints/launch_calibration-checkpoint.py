# -*- coding: utf-8 -*-
"""
Created on Tue May 11 11:19:23 2021

@author: EGO4
"""
import os
import math
import numpy as np

def run_gaze_mapping(root, left_eye_folder, left_eye_depth_path, right_eye_folder, right_eye_depth_path, exp_folder, exp_depth_path, output_folder):

    ############################
    ###### MATLAB ENGINE #######
    ############################
    import matlab.engine
    eng = matlab.engine.start_matlab()
    
    eng.addpath(eng.genpath('/home/vasha/st-bravo_analysis/utils/gaze_map_matlab/'))
    #eng.addpath('/home/vasha/anaconda3/pkgs/gstreamer-1.14.0-hb31296c_0/lib/')
    #end.addpath('/home/vasha/anaconda3/pkgs/gstreamer-1.14.0-hb31296c_0/')
    
    ####################################
    ###### MONOCULAR CALIBRATION #######
    ####################################

    # left_eye_folder = '008'
    # right_eye_folder = '009'
    # exp_folder = '010'
    calibration_folder = left_eye_folder
    calibration_depth_path = left_eye_depth_path

    #ROOT = 'G:/PUPIL_CALIBRATION_DATA/calibration_test_2_noximea/2021_03_22/TEST2/'

    camera_parameters_file = '/home/vasha/st-bravo_analysis/utils/gaze_map_matlab/utils/CameraIntrinsic_D435I_720p_843112073163'
    CALIBRATION_POINTS_NUM = 9.0;
    VIEW = False;

    print('Computing left eye position')
    print(root, left_eye_folder, left_eye_depth_path, 'L', camera_parameters_file, CALIBRATION_POINTS_NUM, output_folder)
    LEFT_EYE_POS  = eng.ComputeEyePos3D(root, left_eye_folder, left_eye_depth_path, 'L', camera_parameters_file, CALIBRATION_POINTS_NUM, output_folder)
    print('Computing right eye position')
    RIGHT_EYE_POS = eng.ComputeEyePos3D(root, right_eye_folder, right_eye_depth_path, 'R', camera_parameters_file, CALIBRATION_POINTS_NUM, output_folder)

    BASELINE = np.sqrt(np.sum(np.power(np.array(LEFT_EYE_POS) - np.array(RIGHT_EYE_POS),2)))
    print('Subject IPD: ' + str(BASELINE))

    ####################################
    ###### BINOCULAR CALIBRATION #######
    ####################################

    print('Computing binocular calibration')
    CALIBRATION_RMSE = eng.BinocularCalibration(output_folder,left_eye_folder, left_eye_depth_path, right_eye_folder, right_eye_depth_path, calibration_folder,CALIBRATION_POINTS_NUM)
    print('Subject IPD: ' + str(BASELINE))

    ################################
    ###### APPLY CALIBRATION #######
    ################################
    CREATE_VIDEO = 1;
    print('Applying binocular calibration')
    FixPoint3D_EXP = eng.Compute3DGaze(output_folder,calibration_folder,exp_folder,exp_depth_path,camera_parameters_file,CREATE_VIDEO, output_folder)

    #TODO: Check for this already being calculated for calibration trials and dont redo if its already done.
    FixPoint3D_LEFT = eng.Compute3DGaze(output_folder,calibration_folder,left_eye_folder,left_eye_depth_path,camera_parameters_file,CREATE_VIDEO, output_folder)

    FixPoint3D_RIGHT = eng.Compute3DGaze(output_folder,calibration_folder,right_eye_folder,right_eye_depth_path,camera_parameters_file,CREATE_VIDEO, output_folder)


    #######################################
    ###### MAP GAZE FROM RS TO XIMEA ######
    #######################################
    print('TODO: Implement mapping gaze to Ximea')







