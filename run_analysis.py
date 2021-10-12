import os
import numpy as np
import cv2
import re
import matplotlib.pyplot as plt
import argparse
import sys
from pathlib import Path
import pandas as pd

#scripts to do the heavy lifting
import utils.bins_to_pngs as bin2png
import utils.timeline as timeline
import utils.tracegen as tracegen
import utils.traceconvert as traceconvert
import stftoolkit as stf
import utils.bag_alignment as bag


def run_analysis(analysis_base_dir,
                 trial_list_path, line_number, 
                 skip_convert_png=True,
                 skip_bag_align=False,
                 skip_calibration=False,
                 skip_fourier=False):
    '''
    Mother script to run analysis of a single trial (line from excel trial list spreadsheet)
    Params:
        analysis_base_dir(str): path for outputting analysis data (/hmet_analysis or /hmet_analysis_2)
        trial_list_path(str): Path to csv file with trial info
        line_number(int): which line/trial to analyze from trial_list
        skip_convert_png(bool): do we need to convert to pngs?
        skip_bag_align(bool): do we need to run the bag depth -> Ximea Alignment?
        skip_calibration(bool): do we need to calculate the calibration?
        skip_convertskip_fourier_png(bool): do we need to run Fourier analysis?
    Outputs:
        Doesn't return anything, but prints a ton of stuff, writes even more stuff to:
            data_path/pngs
            data_path/analysis
            ./output/
    '''

    #parameters that are fixed (note framerates are ideal not necessarily true)
    fps_ximea = 200
    fps_pupil = 200
    fps_depth = 90
    fps_rsrgb = 60
    fps_imu = 200
    resample_fps = 200
    save_batchsize_ximea = 1000
    save_batchsize_depth = 1000
    
    #calibration_types = ['farpoint_eye_cal_pre', 'farpoint_eye_cal_post','left_eye_cal','right_eye_cal', 'val_eye_cal', 'pp_eye_cal', 'color_cal'] #don't have (or need) depth info for farpoint_eye_cal pre and post, pp, or color
    calibration_types = ['left_eye_cal','right_eye_cal', 'val_eye_cal']

    #ximea spec params
    ximea_dims = (1544,2064)
    ximea_horizontal_fov_deg = 61
    
    #Fourier Analysis parameters
    chunk_secs = 2
    chunk_pix = 512
    num_chunks = 500
    cosine_window = True
    spatial_cuttoff_cpd = 14
    temporal_cuttoff_fps = np.inf
    
    #get info from table to get subject name, task, etc.
    print(f'Parsing Trial List line {line_number}....')
    trial_line = pd.read_csv(trial_list_path).iloc[line_number]
    bsdir = trial_line['base_dir']
    date = trial_line['folder']
    subject_name = trial_line['subject']
    trial_num = str(trial_line['trial']).zfill(3)
    task_name = trial_line['task']
    task_iter = str(trial_line['iter'])
    aperature_location = trial_line['aperature_setting'] ####TMP!!!!
    skip_bool = trial_line['skip']
    print(f'This is subject {subject_name}, task {task_name}, iter {task_iter}.')
    if(skip_bool==True):
        print('Skipping this trial because skip boolean is True')
        return()
    
    #put directory structures together for accessing data
    data_folder = os.path.join(bsdir, subject_name, date, str(trial_line['trial']).zfill(3))
    
    #create directory structure to save analyzed files
    ana_folder = os.path.join(analysis_base_dir, subject_name, task_name, task_iter)
    #make paths that dont exist yet
    print(f'Creating analysis directory structure at {ana_folder}.')
    Path(ana_folder).mkdir(parents=True, exist_ok=True)
    
    bin_folder = os.path.join(data_folder,'ximea','ximea')
    png_folder = os.path.join(ana_folder,'pngs')
        
    #Camera Matrices - distortion, intrinsics, and and extrinsics
    print('Loading Camera Matrix Files (Intrinsics/Extrinsics).')
    #Ximea
    camera_intrinsics_folder = f'/home/vasha/st-bravo_analysis/calibration_info/{aperature_location}'
    ximea_distortion = np.loadtxt(os.path.join(camera_intrinsics_folder,'camera2RadialDist.txt'), delimiter=',')
    ximea_distortion = np.array([*ximea_distortion, 0, 0], dtype='float32') #set p1, p2, k3 to zero
    ximea_intrinsics = np.array(np.loadtxt(os.path.join(camera_intrinsics_folder,'Intrinsics_WC.txt'), delimiter=','), dtype='float32')
    #Realsense RGB
    rsrgb_distortion = np.loadtxt(os.path.join(camera_intrinsics_folder,'camera1RadialDist.txt'), delimiter=',')
    rsrgb_distortion = np.array([*rsrgb_distortion, 0, 0], dtype='float32') #set p1, p2, k3 to zero
    rsrgb_intrinsics = np.array(np.loadtxt(os.path.join(camera_intrinsics_folder,'Intrinsics_RS.txt'), delimiter=','), dtype='float32')
    #Extrinsics (ximea to realsense)
    rsrgb_to_ximea_extrinsics_rotation = np.loadtxt(os.path.join(camera_intrinsics_folder,'Rotation_matrix.txt'), delimiter=',')
    rsrgb_to_ximea_extrinsics_rotation = rsrgb_to_ximea_extrinsics_rotation * np.array(((1,-1,-1),(-1,1,-1),(-1,-1,1)))
    rsrgb_to_ximea_extrinsics_translation = np.loadtxt(os.path.join(camera_intrinsics_folder,'Translation_vector.txt'), delimiter=',')
    rsrgb_to_ximea_extrinsics_translation = 1e-3 * rsrgb_to_ximea_extrinsics_translation 
    
    #Run .bin to png conversion for ximea data
    if not skip_convert_png:
        print('Running .bin to .png conversion for Ximea.  This will likely take a few days...')
        
        #run conversion script on binary folder
        print('Skipping bin conversion temporarily, running only calibration folders.')
        #bin2png.convert_trial_directory(bin_folder, png_folder, save_batchsize_ximea, ximea_intrinsics, ximea_distortion)
        print(f'Finished .bin to .png conversion for {ana_folder}.')
        
        #run conversion .bin to .png for calibrations if it hasn't been done already.
        if(not trial_line['subject'] in ['bu']):
            print('This is a human trial with calibrations. Searching for corresponding calibration pngs')
            for caltype in calibration_types:
                #create unique identifier for calibration called calib_id
                folderid = trial_line[caltype]
                calib_id = f'{date}_{folderid}' #can't use calib type in name because pre/post are shared for far point
                calibration_png_folder = os.path.join(analysis_base_dir, subject_name, 'calib', calib_id, 'pngs')
#                 if(Path(os.path.join(calibration_png_folder,'frame_0.png')).exists()):
#                     print(f'Found PNG Calibration folder for corresponding {caltype}.')
#                 else:
                    #print(f'Did not find corresponding PNG calibration for {caltype}. Creating now at {calibration_png_folder}')
                print(f'Creating PNGs for calibration for {caltype}: {calibration_png_folder}')
                calibration_bin_folder = os.path.join(bsdir, subject_name, date, str(trial_line[caltype]).zfill(3),'ximea','ximea')
                bin2png.convert_trial_directory(calibration_bin_folder, calibration_png_folder, save_batchsize_ximea, ximea_intrinsics, ximea_distortion)
        else:
            print('This is a manniquen (buddy) trial, no need to convert calibration trial pngs.')       

    else:
        print('Skipping .bin to .png conversion for Ximea.')
        
    # align depth -> RGB & Ximea space using realsense align_to function and .bag files
    if not skip_bag_align:
        print('Running Depth Alignment to RGB & Ximea Space. This will take a few hours....')
        print(f'Data folder is: {data_folder}')
        print(f'Analysis folder is: {ana_folder}')  
        bag.create_aligned_depth_files(recording_folder=data_folder,
                               output_folder=ana_folder,
                               ximea_distortion=ximea_distortion, 
                               ximea_intrinsics=ximea_intrinsics, 
                               rgb_distortion=rsrgb_distortion, 
                               rgb_intrinsics=rsrgb_intrinsics,
                               rgb_to_ximea_rotation=rsrgb_to_ximea_extrinsics_rotation,
                               rgb_to_ximea_translation=rsrgb_to_ximea_extrinsics_translation,
                                       bag_in_path=f'/home/vasha/st-bravo_analysis/bag/sample_final.bag'
                                       #bag_in_path=f'/home/vasha/st-bravo_analysis/bag/sample_final-Copy{line_number}.bag'
                              )
        
        #align depth -> RGB & Ximea for calibrations if it hasn't been done already.
        if(not trial_line['subject'] in ['bu']):
            print('This is a human trial with calibrations. Searching for corresponding calibration pngs')
            for caltype in calibration_types:
                #create unique identifier for calibration called calib_id
                folderid = trial_line[caltype]
                calib_id = f'{date}_{folderid}' #can't use calib type in name because pre/post are shared for far point
                calibration_raw_folder = os.path.join(bsdir, subject_name, date, str(trial_line[caltype]).zfill(3))
                calibration_ana_folder = os.path.join(analysis_base_dir, subject_name, 'calib', calib_id)
                calibration_bag_file = os.path.join(calibration_ana_folder, 'depth_ximea.bag')
                print(f'Creating Calibration Bag File for {caltype}: {calibration_bag_file}')
                print(calibration_raw_folder)
                print(calibration_ana_folder)                    
                bag.create_aligned_depth_files(recording_folder=calibration_raw_folder,
                           output_folder=calibration_ana_folder,
                           ximea_distortion=ximea_distortion, 
                           ximea_intrinsics=ximea_intrinsics, 
                           rgb_distortion=rsrgb_distortion,
                           rgb_intrinsics=rsrgb_intrinsics,
                           rgb_to_ximea_rotation=rsrgb_to_ximea_extrinsics_rotation,
                           rgb_to_ximea_translation=rsrgb_to_ximea_extrinsics_translation,
                                               bag_in_path=f'/home/vasha/st-bravo_analysis/bag/sample_final-Copy{line_number}.bag'
                          )
                    
                    
        else:
            print('This is a manniquen (buddy) trial, no need to convert calibration trial pngs.')       
        
    else:
        print('Skipping Depth Alignment to RGB & Ximea Space')

    # run eye tracking calibration unless we want to skip OR is a manniquen (buddy) trial
    if (not skip_calibration) or (not trial_line['subject'] in ['bu']):
        print('Running gaze point localization based on matching calibration trial....')
    else:
        print('Skipping Eye Tracking & Calibration Analysis')
        
    # run fourier analysis (Power Spectrum Calculation)    
    if not skip_fourier:
        print(f'Running Fourier Analaysis for {num_chunks} chunks of size {chunk_pix} pixels....')

        #some calculations for Fourier Analysis
        chunk_frames = int(chunk_secs*resample_fps)
        ximea_horizontal_ppd = ximea_dims[1]/ximea_horizontal_fov_deg
        ppd = ximea_horizontal_ppd
    else:
        print('Skipping Fourier Analysis')
        
    print(f'All Done with analysis for subject: {subject_name}, task: {task_name}, iter: {task_iter}!')
    

def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--analysis_path", help="path to analysis (output of this script)", type=str, 
                        default='/hmet_analysis')
    parser.add_argument("-t","--trial_list_path", help="path to csv file containing trial info", type=str, 
                        default='~/st-bravo_analysis/trial_list.csv')
    parser.add_argument("-l", "--line_number", help="line number in trial_list to analyze", type=int)
    parser.add_argument("-p", "--skip_convert_png", help="skip convert ximea .bin to pngs", type=bool, default=True)
    parser.add_argument("-b", "--skip_bag_align", help="skip bag alignment", type=bool, default=False)
    parser.add_argument("-c", "--skip_calibration", help="skip eye tracking calibration", type=bool, default=False)
    parser.add_argument("-f", "--skip_fourier", help="skip fourier analysis", type=bool, default=False)
    #parser.add_argument("-s", "--stop_time", help="time to stop analysis")
   
    
    args = parser.parse_args()
    print(f'analyzing line {args.line_number} of {args.trial_list_path}')

    #launch analysis
    run_analysis(args.analysis_path, 
                 args.trial_list_path, args.line_number, 
                 args.skip_convert_png, args.skip_bag_align, 
                 args.skip_calibration, args.skip_fourier)

    
if __name__ == "__main__":
   main(sys.argv[1:])