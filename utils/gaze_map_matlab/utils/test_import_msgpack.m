addpath utils\

folder = 'C:\Users\EGO4\recordings\2020_06_01\001';
folder = 'G:\PUPIL_CALIBRATION_DATA\3D_EYE_POS_AND_GAZE\TEST_0\LE';
import_msgpack(folder)

fileID = fopen([folder '\offline_data\reference_locations.txt']);
C = textscan(fileID,formatSpec)