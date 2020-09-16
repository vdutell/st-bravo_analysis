import os
import numpy as np
import cv2
import re
import matplotlib.pyplot as plt
import argparse
import sys
from pathlib import Path
import pandas as pd

#scripts to do the dirty work
import utils.bins_to_pngs as bin2png
import utils.timeline as timeline
import utils.tracegen as tracegen
import utils.traceconvert as traceconvert
import stftoolkit as stf

def run_analysis(base_dir, trial_list_path, line_number, convert_png=False):
    '''
    Mother script to run analysis of a single trial
    Params:
        base_dir(str): path with data 
        trial_list_path(str): Path to csv file with trial info
        line_number(int): which line/trial to analyze from trial_list
        convert_png(bool):do we need to convert to pngs?
    Outputs:
        Doesn't return anything, but prints a ton of stuff, writes even more stuff to:
            data_path/pngs
            data_path/analysis
            ./output/
    '''
    #use precomputed conversion to ximea coordinates for pupil positions where values are individually adjusted for proper alignment
    run_rs2xi=False
    
    #parameters that are fixed
    fps_ximea = 200
    fps_pupil = 200
    fps_depth = 90
    fps_rsrgb = 30
    #fps_imu = ???
    resample_fps = 200
    
    #ximea speec params
    img_dims = (1544,2064)
    horizontal_fov_deg = 61.
    
    #pupil positions parameters
    degrees_eccentricity_h=20
    degrees_eccentricity_v=15
    
    #Fourier Analysis parameters
    chunk_secs = 2 #2
    chunk_pix = 512
    num_chunks = 100
    cosine_window=True
    spatial_cuttoff_cpd = 14
    temporal_cuttoff_fps = np.inf
    
    #some calculations
    chunk_frames = int(chunk_secs*resample_fps)
    horizontal_ppd = img_dims[1]/horizontal_fov_deg
    ppd = horizontal_ppd

    #get names of folders and files we'll use a lot
    data_dir = os.path.join(base_dir, 'raw')
    png_dir = os.path.join(base_dir, 'pngs')
    ana_dir = os.path.join(base_dir, 'analysis')
    trace_dir = os.path.join(base_dir, 'traces')
    
    #get info from table to get individual_task_folders
    trial_line = pd.read_csv(trial_list_path).iloc[line_number]
    data_folder = os.path.join(data_dir, trial_line['folder'], str(trial_line['trial']).zfill(3))
    png_folder = os.path.join(png_dir, trial_line['task'], trial_line['subject'], str(trial_line['iter']))
    ana_folder = os.path.join(ana_dir, trial_line['task'], trial_line['subject'], str(trial_line['iter']))
    #make paths that dont exist yet
    Path(png_folder).mkdir(parents=True, exist_ok=True)
    Path(ana_folder).mkdir(parents=True, exist_ok=True)
    
    #flag this as a human or buddy (fixed camera) trial
    if(trial_line['subject'] in ['bu']):
        has_fixations = False
    else:
        has_fixations = True
        
    #flag for save size
    if(trial_line['folder'] in ['2020_02_26']):
        save_batchsize = 4000
    elif(trial_line['folder'] in ['2020_02_27']):
        save_batchsize = 400
    else:
        print('Dont know what save batchsize is. Add date to trial_line[folder] check. Error for you coming up shortly...')
        
    #trace types
    if(has_fixations):
        #trace_types = ['fixed_central', 'fixed_rand_start', 'foveal','periph_l','periph_r','periph_u','periph_d']
        trace_types = ['fixed_central', 'foveal', 'fixed_rand_start']
    else:
        trace_types = ['fixed_central', 'fixed_rand_start']
    
    #step one - convert from binary files to pngs
    #(commenting right now because we've done this for the current dataset)
    print(f'\n***Step 1: Converting Trial Directory from Binary to PNG for {data_folder} to {png_folder}. This will take a long time.....')
    if(convert_png):
        bin2png.convert_trial_directory(data_dir, png_dir, trial_list_path, line_number, save_batchsize)
    else:
        print('skipping conversion for now (its probably aready been done)')
    
    #step two - convert ximea timestamps to unix time for consistent timeline between devices
    print(f'\n***Step 2: Resampling Timeline to deal with dropped frames...')
    timestamp_file = os.path.join(data_folder, 'ximea', 'timestamps_ximea.tsv')
    timesync_file = os.path.join(data_folder, 'ximea', 'timestamp_camsync_ximea.tsv')
    ximea_timestamps = timeline.convert_ximea_time_to_unix_time(timestamp_file, timesync_file)
    np.save(os.path.join(ana_folder, 'ximea_timestamps.npy'), ximea_timestamps)
    np.savetxt(os.path.join(ana_folder, 'ximea_timestamps.csv'), ximea_timestamps, header='', delimiter=',', fmt='%f')

    #step three - match all devices to a common timeline and resample to closest samples to timepoint
    print(f'\n***Step 3: Sanity Checking Recorded Timestamps for Overlap & Consistency...')
    #Note: Need to add code to do this for IMU data also 
    #XIMEA
    #we already have the converted ximea timestamps, get pupil labs timeline
    #ximea_timestamps = np.load(os.path.join(ana_folder, 'world_timestamps.npy'))
    #PUPIL
    if(has_fixations):
        pupil_data_fname = os.path.join(ana_dir, 'calibrated_traces', f"fixation_point{trial_line['folder']}-{str(trial_line['trial']).zfill(3)}.txt")
        pupil_data = np.loadtxt(pupil_data_fname,delimiter=',')
        pupil_timestamps = pupil_data[:,0]
    #DEPTH
    depth_timestamps_fname = os.path.join(data_folder, 'depth', 'timestamps.csv')
    depth_timestamps = np.loadtxt(depth_timestamps_fname, delimiter=',', skiprows=0)
    #Realsense RGB
    rsrgb_timestamps_fname = os.path.join(data_folder, 'world_timestamps.npy')
    rsrgb_timestamps = np.load(rsrgb_timestamps_fname)
    #IMUs
    #imu_timestamps = LOAD THE IMU DATA HERE
    #make sure there is a significant overlap.
    if(has_fixations):
        timeline.calc_olap_fraction([ximea_timestamps[:,1], pupil_timestamps, depth_timestamps, rsrgb_timestamps])
        timeline.calc_dropped_frames(pupil_timestamps, fps_pupil, 'Pupil')
    else:
        timeline.calc_olap_fraction([ximea_timestamps[:,1], depth_timestamps, rsrgb_timestamps])
    # add IMU timestamps here
    #calculate number of dropped frames for each timestamp
    timeline.calc_dropped_frames(ximea_timestamps[:,1], fps_ximea, 'Ximea')
    timeline.calc_dropped_frames(depth_timestamps, fps_depth, 'Depth')
    timeline.calc_dropped_frames(rsrgb_timestamps, fps_rsrgb, 'RSRGB')
    
    print(f'\n***Step 4: Matching Samples for all sensors to a Common & Consistent timeline...')
    if(has_fixations):
        common_timeline, sampleidx_list = timeline.assign_common_timeline([ximea_timestamps[:,1], pupil_timestamps, depth_timestamps, rsrgb_timestamps],
                                                                          resample_fps, starttime=trial_line['start_sec'], endtime=trial_line['end_sec'])
    else:
        common_timeline, sampleidx_list = timeline.assign_common_timeline([ximea_timestamps[:,1], depth_timestamps, rsrgb_timestamps],
                                                                          resample_fps, starttime=trial_line['start_sec'], endtime=trial_line['end_sec'])
    common_timeline_zstart = common_timeline - common_timeline[0]
    
    #save variables with common timeline
    #common_timeline = common_timeline
    ximea_frame_idx = sampleidx_list[0]
    if(has_fixations):
        pupil_samples_idx = sampleidx_list[1]
        depth_samples_idx = sampleidx_list[2]
        rsrgb_samples_idx = sampleidx_list[3]
    else:
        depth_samples_idx = sampleidx_list[1]
        rsrgb_samples_idx = sampleidx_list[2] 
    #include IMU data later
    
    
    #uncomment this to recalculate
#     #save values
#     np.savetxt(os.path.join(ana_folder, 'common_timeline.csv'), common_timeline, header='', delimiter=',', fmt='%f')
#     np.savetxt(os.path.join(ana_folder, 'common_timeline_zstart.csv'), common_timeline_zstart, header='', delimiter=',', fmt='%f') #timeline that starts at zero.
#     np.savetxt(os.path.join(ana_folder, 'common_timeline_ximea_sampleidx.csv'), ximea_frame_idx, header='', delimiter=',', fmt='%i')
#     np.savetxt(os.path.join(ana_folder, 'common_timeline_depth_sampleidx.csv'), depth_samples_idx, header='', delimiter=',', fmt='%i')
#     np.savetxt(os.path.join(ana_folder, 'common_timeline_rsrgb_sampleidx.csv'), rsrgb_samples_idx, header='', delimiter=',', fmt='%i')
#     if(has_fixations):
#         np.savetxt(os.path.join(ana_folder, 'common_timeline_pupil_sampleidx.csv'), pupil_samples_idx, header='', delimiter=',', fmt='%i')




        #for pupil,(IMU later), but NOT ximea or depth - we can use the resampled data directly to avoid introducing bugs
        #print(f'Pupil Data Shape Before Resample: {pupil_data.shape}')
        #pupil_data = pupil_data.take(pupil_samples_idx,axis=0) this is done inside pupil_conversion manual
        #print(f'Pupil Data Shape Post Resample: {pupil_data.shape}')

    if(has_fixations):
        pupil_positions_converted_file = os.path.join(ana_folder,'pupil_positions_ximea_space.npy')
        if(run_rs2xi):
            print(f'\n***Step 5: L/R Eye processing and Convert fixation positions Realsense to Ximea...')
            # pupil positions file columns are leftx, lefy, rightx, righty, binocularx, binoculary
            #get binocular data (has been weighted between confidence in left and right eye)
            pupil_positions = pupil_data[:,[5,6]]
            #print(pupil_positions.shape)
            #pupil_positions = traceconvert.process_lr_trace_criterion(pupil_positions, trial_line['trace_key']) #average the left and right eye positions in x and y
            pupil_positions = traceconvert.convert_positions_rsrgb_ximea(png_folder,data_folder,ana_folder,pupil_positions,
                                                              ximea_frame_idx, rsrgb_samples_idx)
            np.save(pupil_positions_converted_file,pupil_positions)
            
        else:
            pupil_positions = np.load(pupil_positions_converted_file)
    else:
        print('Skipping to step 6 because we dont have fixations for buddy trial')
        pupil_positions=None
    
    print(f'\n***Step 6: Fourier Chunking and Overlay Eye Motion...')
    #VALIDATION_TSTART = 20000 #use this to specify a start time
    VALIDATION_TSTART = None
    
    for trace_type in trace_types:
        print(f'Running {trace_type}...')
            
        #choose value for eccentricity
        if(trace_type in ['fixed_central','fixed_rand_start','foveal']):
            degrees_ecc = None
        elif(trace_type in ['periph_l','periph_r']):
             degrees_ecc = degrees_eccentricity_h
        elif(trace_type in ['periph_u','periph_d']):
            degrees_ecc = degrees_eccentricity_v
                
        ############## SAVE A SINGLE EXAMPLE (this is also a cheat to get dimensions) #############
        print(f'Saving a single Exemplar to {ana_folder}')
        #generate a trace
        trace_start_idx, trace_lcorner_xy = tracegen.generate_trace(trace_type=trace_type,
                                                                    chunk_samples=chunk_frames,
                                                                    chunk_dim=chunk_pix, frame_dims=img_dims,
                                                                    timeline_len=len(common_timeline),
                                                                    pupil_positions=pupil_positions, ppd=ppd,
                                                                    degrees_eccentricity=degrees_ecc,
                                                                    validation_tstart=VALIDATION_TSTART)
        #THIS NEXT STEP IS KEY, AND VERY SUBTLE. XIMEA FRAMES NEED TO BE RESAMPLED
        #to be specific, the trace start index is in common timeline indices.
        #ximea raw frame numbers (frame_f.png) must be inferred from the common timeline index that has been best matched.
        #in other words, to get the ximea frame that happens at index i in our common timeline, 
        #we need to get the number f stored at ximea_frame_idx[i]. The pixels values for the closest frame are stored at 'frame_f.png'
        #We did this earlier to our pupil data with the call pupil_data = pupil_data.take(pupil_samples_idx,axis=0)
        #print(ximea_frame_idx[:500])
        chunk_frame_indices = ximea_frame_idx[trace_start_idx:trace_start_idx+chunk_frames]
        #pull out the movie chunk corresponding to this trace
        movie_chunk = np.zeros((chunk_frames, chunk_pix, chunk_pix, 3))
        movie_chunk_norm = np.zeros((chunk_frames, chunk_pix, chunk_pix, 3)) #for visualization
        for i, f in enumerate(chunk_frame_indices):
            print('*',end='')
            frame = cv2.imread(os.path.join(png_folder,f'frame_{f}.png'))
            chunk = frame[trace_lcorner_xy[i,1]:trace_lcorner_xy[i,1]+chunk_pix, 
                                   trace_lcorner_xy[i,0]:trace_lcorner_xy[i,0]+chunk_pix]
            chunk_norm = chunk.copy() #make sure we don't change our original chunk
            chunk_norm = traceconvert.adjust_gamma(chunk_norm,gamma=3)
            movie_chunk[i] = chunk
            movie_chunk_norm[i] = chunk_norm
            
        #write movie
        fourcc = cv2.VideoWriter_fourcc(*'MPEG')
        video = cv2.VideoWriter(os.path.join(ana_folder, f'ExampleChunkVideo_{trace_type}_color.avi'), fourcc, 200, (chunk_pix,chunk_pix))
        for i in range(np.shape(movie_chunk_norm)[0]):
            video.write(np.uint8(movie_chunk_norm[i]))
        video.release()
            
        #get fourier transform of that trace & save output
        ps_3d, ps_2ds, fqs_space, fqs_time = stf.st_ps(movie_chunk, ppd, resample_fps,
                                                      cosine_window=cosine_window, rm_dc=True, use_cupy_fft=False) 
        ps_2d_all, ps_2d_vert, ps_2d_horiz, ps_2d_l_diag, ps_2d_r_diag = ps_2ds

        #save example of each raw data
        np.save(os.path.join(ana_folder, f'Example3dPowerSpec_{trace_type}.npy'), ps_3d)
        np.save(os.path.join(ana_folder, f'Example2dAllPowerSpec_{trace_type}.npy'), ps_2d_all)
        np.save(os.path.join(ana_folder, f'Example2dVertPowerSpec_{trace_type}.npy'), ps_2d_vert)
        np.save(os.path.join(ana_folder, f'Example2dHorizPowerSpec_{trace_type}.npy'), ps_2d_horiz)
        np.save(os.path.join(ana_folder, f'Example2dLDiagPowerSpec_{trace_type}.npy'), ps_2d_l_diag)
        np.save(os.path.join(ana_folder, f'Example2dRDiagPowerSpec_{trace_type}.npy'), ps_2d_r_diag)
        np.save(os.path.join(ana_folder, f'Example3dPowerSpecFreqsSpace_{trace_type}.npy'), fqs_space)
        np.save(os.path.join(ana_folder, f'Example3dPowerSpecFreqsTime_{trace_type}.npy'), fqs_time)
        #save example of each plot
        stf.da_plot_power(ps_2d_all, fqs_space, fqs_time, show_onef_line=True, logscale=True,
                  figname=f'ExampleAllPowerSpec_{num_chunks}_chunks_{trace_type}', 
                  saveloc=ana_folder, grey_contour=False)
        stf.da_plot_power(ps_2d_vert, fqs_space, fqs_time, show_onef_line=True, logscale=True,
                  figname=f'ExampleVertPowerSpec_{num_chunks}_chunks_{trace_type}', 
                  saveloc=ana_folder, grey_contour=False)
        stf.da_plot_power(ps_2d_horiz, fqs_space, fqs_time, show_onef_line=True, logscale=True,
                  figname=f'ExampleHorizPowerSpec_{num_chunks}_chunks_{trace_type}', 
                  saveloc=ana_folder, grey_contour=False)
        stf.da_plot_power(ps_2d_l_diag, fqs_space, fqs_time, show_onef_line=True, logscale=True,
                  figname=f'ExampleLDiagPowerSpec_{num_chunks}_chunks_{trace_type}', 
                  saveloc=ana_folder, grey_contour=False)        
        stf.da_plot_power(ps_2d_r_diag, fqs_space, fqs_time, show_onef_line=True, logscale=True,
                  figname=f'ExampleRDiagPowerSpec_{num_chunks}_chunks_{trace_type}', 
                  saveloc=ana_folder, grey_contour=False)
        
        ################## NOW LOOP OVER MANY Examples and take their mean         #####################
        print(f'Taking Mean of {num_chunks} traces of type {trace_type}...')
        #store mean values in arrays
        ps_3d_mean = np.zeros_like(ps_3d)
        ps_2d_all_mean = np.zeros_like(ps_2d_all)
        ps_2d_vert_mean = np.zeros_like(ps_2d_vert)
        ps_2d_horiz_mean = np.zeros_like(ps_2d_horiz)
        ps_2d_l_diag_mean = np.zeros_like(ps_2d_l_diag)
        ps_2d_r_diag_mean = np.zeros_like(ps_2d_r_diag)
        
        for i in range(num_chunks):
            
            trace_start_idx, trace_lcorner_xy = tracegen.generate_trace(trace_type=trace_type,
                                                                        chunk_samples=chunk_frames,
                                                                        chunk_dim=chunk_pix, frame_dims=img_dims,
                                                                        timeline_len=len(common_timeline),
                                                                        pupil_positions=pupil_positions, ppd=ppd,
                                                                        degrees_eccentricity=degrees_ecc,
                                                                        validation_tstart=VALIDATION_TSTART)
            chunk_frame_indices = ximea_frame_idx[trace_start_idx:trace_start_idx+chunk_frames]
            #pull out the movie chunk corresponding to this trace
            movie_chunk = np.zeros((chunk_frames, chunk_pix, chunk_pix, 3))
            for i, f in enumerate(chunk_frame_indices):
                print('*',end='')
                frame = cv2.imread(os.path.join(png_folder,f'frame_{f}.png'))
                chunk = frame[trace_lcorner_xy[i,1]:trace_lcorner_xy[i,1]+chunk_pix, 
                                       trace_lcorner_xy[i,0]:trace_lcorner_xy[i,0]+chunk_pix]
                movie_chunk[i] = chunk    
            #get fourier transform of that trace & save output
            ps_3d, ps_2ds, fqs_space, fqs_time = stf.st_ps(movie_chunk, ppd, resample_fps,
                                                        cosine_window=cosine_window, rm_dc=True, use_cupy_fft=False)
            ps_2d_all, ps_2d_vert, ps_2d_horiz, ps_2d_l_diag, ps_2d_r_diag = ps_2ds

            
            ps_3d_mean += ps_3d
            ps_2d_all_mean += ps_2d_all
            ps_2d_vert_mean += ps_2d_vert
            ps_2d_horiz_mean += ps_2d_horiz
            ps_2d_l_diag_mean += ps_2d_l_diag
            ps_2d_r_diag_mean += ps_2d_r_diag
        
        print(f'Finished Calculating Power Spectra for Trace Type {trace_type}. Now saving...')
        np.save(os.path.join(ana_folder, f'Mean3dPowerSpec_{trace_type}.npy'), ps_3d_mean)
        np.save(os.path.join(ana_folder, f'Mean2dAllPowerSpec_{trace_type}.npy'), ps_2d_all_mean)
        np.save(os.path.join(ana_folder, f'Mean2dVertPowerSpec_{trace_type}.npy'), ps_2d_vert_mean)
        np.save(os.path.join(ana_folder, f'Mean2dHorizPowerSpec_{trace_type}.npy'), ps_2d_horiz_mean)
        np.save(os.path.join(ana_folder, f'Mean2dLDiagPowerSpec_{trace_type}.npy'), ps_2d_l_diag_mean)
        np.save(os.path.join(ana_folder, f'Mean2dRDiagPowerSpec_{trace_type}.npy'), ps_2d_r_diag_mean)
        np.save(os.path.join(ana_folder, f'Mean3dPowerSpecFreqsSpace_{trace_type}.npy'), fqs_space)
        np.save(os.path.join(ana_folder, f'Mean3dPowerSpecFreqsTime_{trace_type}.npy'), fqs_time)    
        
        #plot a bit
        stf.da_plot_power(ps_2d_all_mean, fqs_space, fqs_time, show_onef_line=True, logscale=True,
                  figname=f'MeanAllPowerSpec_{num_chunks}_chunks_{trace_type}', 
                  saveloc=ana_folder, grey_contour=False)
        stf.da_plot_power(ps_2d_vert_mean, fqs_space, fqs_time, show_onef_line=True, logscale=True,
                  figname=f'MeanVertPowerSpec_{num_chunks}_chunks_{trace_type}', 
                  saveloc=ana_folder, grey_contour=False)
        stf.da_plot_power(ps_2d_horiz_mean, fqs_space, fqs_time, show_onef_line=True, logscale=True,
                  figname=f'MeanHorizPowerSpec_{num_chunks}_chunks_{trace_type}', 
                  saveloc=ana_folder, grey_contour=False)
        stf.da_plot_power(ps_2d_l_diag_mean, fqs_space, fqs_time, show_onef_line=True, logscale=True,
                  figname=f'MeanLDiagPowerSpec_{num_chunks}_chunks_{trace_type}', 
                  saveloc=ana_folder, grey_contour=False)
        stf.da_plot_power(ps_2d_r_diag_mean, fqs_space, fqs_time, show_onef_line=True, logscale=True,
                  figname=f'MeanRDiagPowerSpec_{num_chunks}_chunks_{trace_type}', 
                  saveloc=ana_folder, grey_contour=False)
    
    
    print(f'All Done with this trial!')
                
            
    
def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--data_path", help="path to data")
    parser.add_argument("-t","--trial_list_path", help="path to csv file containing trial info", type=str, 
                        default='~/st-bravo_analysis/trial_list.csv')
    parser.add_argument("-l", "--line_number", help="line number in trial_list to analyze", type=int, default=0)
    parser.add_argument("-c", "--convert_png", help="also convert pngs", type=bool, default=False)
    #parser.add_argument("-s", "--stop_time", help="time to stop analysis")
   
    
    args = parser.parse_args()
    print(f'analyzing line {args.line_number} of {args.trial_list_path} for data in {args.data_path}')

    #launch analysis
    run_analysis(args.data_path, args.trial_list_path, args.line_number, args.convert_png)

    
if __name__ == "__main__":
   main(sys.argv[1:])


# def convert_bin_pngs(filename, first_fnum save_folder, save_batchsize, dims=(1544,2064), img_format='XI_RAW8'):
#     '''
#     Take a file saved in .bin format from a ximea camera, and convert it to png images.
#     Parameters:
#         filename (str): file to be converted
#         save_folder (str): folder to save png files
#         im_shape (2pule ints): shape of image
#         img_format (str): Image format files are saved
#     Returns:
#         None
#     '''
    

#     for i in range(first_fnum, first_fnum+save_batchsize):
#         save_filepath = os.path.join(save_folder, 'frame_{first_fnum + '.png')
#     nbytes = np.prod(dims)
        
#     elif(img_format=='XI_RAW8'):
#         with open(filename, 'rb') as f:
#             bs = f.read(1)
#             while(bs):
#                 binary_img = []
#                 bs = f.read(1)
#                 bs = int.from_bytes(bs,'big')
#                 binary_img.append(bs)
#             f.close()
#         im = np.array(binary_img).reshape(im_shape)
#         #note this uses bilinear interpolation
#         im = cv2.cvtColor(np.uint8(im), cv2.COLOR_BayerGR2RGB)
#         im = im.astype(np.uint8)
        
#     cv2.imwrite(save_filepath, im)
#     print('*',end='')
    
#     return()


# def bin_to_im(binfile, nframe, dims=(1544,2064)):
#     '''
#     convert a single image from 8-bit raw bytes to png image.
#     Input:
#         binfile (str): path to binary file
#         dims (2ple int): What are the dimensions of the iamge?
#         nframe (int): Which frame number do we want within image?
#         '''
#     a = []
#     # for uint8
#     nbytes = np.prod(dims)
#     startbyte = nframe*nbytes
#     with open(binfile, 'rb') as fn:
#         fn.seek(startbyte)
#         bs = fn.read(1)
#         for i in range(nbytes):
#             bs = fn.read(1)
#             bs = int.from_bytes(bs,'big')
#             a.append(bs)
            
#     a = np.array(a)
#     im = a.reshape(dims)
#     imc = cv2.cvtColor(np.uint8(im), cv2.COLOR_BayerGR2RGB)
    
#     return(imc)

# def convert_folder(read_folder, write_folder):
#     '''
#     Convert a folder of raw .bin files to .pngs
#     Params:
#         read_folder (str): where are .bin files stored?
#         write_folder (str): where should we write the pngs to?
#     '''
#     #loop through files in folder
#     for f in os.listdir(read_folder):
#         if f.endswith(".bin"):
#             convert_bin_png(os.path.join(read_folder,f), write_folder)
            
# def calc_timestamp_stats(timestamp_file, write_folder):
#     '''
#       Figure out how well we did with timing in terms of capturing images
#       Params:
#           timestamp_file (str): tsv file holding timestamp data
#           write_folder (str): folder to store output stats
#     '''
#     with open(timestamp_file, 'r') as f:
#         ts_table=list(zip(line.strip().split('\t') for line in f))
# #        print(ts_table)
# #         line = f.readline() #column headers
# #         while(line):
# #             line = f.readline()
# #             print(line)
# #             frame, os, od = [re.split(line.strip(), '\t')]
# #             print(frame, os, od)
#         f.close()
    
#     ts_table = np.squeeze(np.array(ts_table[1:]).astype('float'))
#     #calculate dts between frames
#     lr_camera_dcaps = np.abs(ts_table[:,1] - ts_table[:,2])
#     os_dts = ts_table[1:,2] - ts_table[:-1,2]
#     od_dts = ts_table[1:,4] - ts_table[:-1,4]
#     cy_dts = ts_table[1:,6] - ts_table[:-1,6]
#     #calculate frame skips
#     os_skips = (ts_table[1:,1] - ts_table[:-1,1]) - 1
#     od_skips = (ts_table[1:,3] - ts_table[:-1,3]) - 1
#     cy_skips = (ts_table[1:,5] - ts_table[:-1,5]) - 1
    
#     print(f'Mean camera time disparity: {np.mean(lr_camera_dcaps):.4f} seconds')
#     print(f'Mean OS dts: {np.mean(os_dts):.4f} seconds')
#     print(f'Mean OD dts: {np.mean(od_dts):.4f} seconds')
#     print(f'Mean CY dts: {np.mean(cy_dts):.4f} seconds')    
    
#     print(f'Mean OS skips: {np.mean(os_skips):.2f} frames')
#     print(f'Mean OD skips: {np.mean(od_skips):.2f} frames')
#     print(f'Mean CY skips: {np.mean(cy_skips):.2f} frames')   
    
#     plt.hist(os_dts, label = 'OS dt', alpha=0.6, bins=30);
#     plt.hist(od_dts, label = 'OD dt', alpha=0.6, bins=30);
#     plt.hist(cy_dts, label = 'CY dt', alpha=0.6, bins=30);
#     plt.axvline(1/200, label='200fps')
#     plt.legend()
#     plt.ylabel('Seconds')
#     plt.title('Within CameraTiming Disparity for World Camera')
#     plt.savefig(os.path.join(write_folder,'timestamp_stats_within_cam.png'))
#     plt.show()
   
#     plt.hist(lr_camera_dcaps, label = 'OD/OS Disparity', alpha=0.6, bins=30);
#     plt.legend()
#     plt.ylabel('Seconds')
#     plt.title('Between Camera Timing Disparity for World Camera')
#     plt.savefig(os.path.join(write_folder,'timestamp_stats_between_cam.png'))
#     plt.show()
    
#     plt.plot(os_dts,label='OS dt')
#     plt.plot(od_dts,label='OD dt')
#     plt.plot(cy_dts,label='CY dt')
#     plt.axhline(1/200, label='200fps')
#     plt.xlabel('frame number')
#     plt.ylabel('DT')
#     plt.legend()
#     plt.title('DT Over Collection')
#     plt.savefig(os.path.join(write_folder,'dt_over_collection.png'))
#     plt.show()
    
#     plt.plot(os_skips,label='OS skips')
#     plt.plot(od_skips,label='OD skips')
#     plt.plot(cy_skips,label='CY skips')
#     plt.axhline(0, label='Zero')
#     plt.xlabel('capture number')
#     plt.ylabel('Skipped Frames')
#     plt.legend()
#     plt.title('Skipped Frames Over Collection')
#     plt.savefig(os.path.join(write_folder,'skipped_frames_over_collection.png'))
#     plt.show()
    
# def run_ximea_analysis(capture_folder, analysis_folder, timestamp_stats=True, convert_ims=True):
#     '''
#     Analyze video data, including converting .bin files to png files.
#     '''

#     try:
        
#         #calcuate stats on frame capture
#         if(timestamp_stats):
#             calc_timestamp_stats(os.path.join(capture_folder,'timestamps.tsv'),
#                                 analysis_folder)
        
#         if(convert_ims):
 
#             #OD
#             od_cap_folder = os.path.join(capture_folder,'cam_od')
#             od_ana_folder = os.path.join(analysis_folder,'cam_od')
#             if not os.path.exists(od_ana_folder):
#                 os.makedirs(od_ana_folder)    
#             #convert_folder(od_cap_folder, od_ana_folder)
#             od_save_thread = Process(target=convert_folder, args=(od_cap_folder, od_ana_folder))
#             od_save_thread.daemon = True
#             od_save_thread.start()  
            
#             #OS
#             os_cap_folder = os.path.join(capture_folder,'cam_os')
#             os_ana_folder = os.path.join(analysis_folder,'cam_os')
#             if not os.path.exists(os_ana_folder):
#                 os.makedirs(os_ana_folder)
#             #convert_folder(os_cap_folder, os_ana_folder)
#             os_save_thread = Process(target=convert_folder, args=(os_cap_folder, os_ana_folder))
#             os_save_thread.daemon = True
#             os_save_thread.start()  

#             #same for CY
#             cy_cap_folder = os.path.join(capture_folder,'cam_cy')
#             cy_ana_folder = os.path.join(analysis_folder,'cam_cy')
#             if not os.path.exists(cy_ana_folder):
#                 os.makedirs(cy_ana_folder)
#             #convert_folder(cy_cap_folder, cy_ana_folder)
#             cy_save_thread = Process(target=convert_folder, args=(cy_cap_folder, cy_ana_folder))
#             cy_save_thread.daemon = True
#             cy_save_thread.start() 
            
#             #wait for all processes to finish
#             print('Waiting for frame conversions...')
#             for proc in [od_save_thread, os_save_thread, cy_save_thread]:
#                 proc.join()
#             print('Done with frame conversions.')
            
#     except Exception as e:
#         print(e)
#         print('Problem with analzing saved scene camera files. Tell Vasha to make more informative error reporting!')

# def run_analysis(subject_name=None, task_name=None, exp_type=None, 
#                  read_dir='./capture', save_dir='./analysis',
#                  run_timestamp_stats=True, run_convert_ims=True):
    
#     '''
#     Run a data analysis, on a pre or post calibration, or an experiment.
#     Params:
#         subject (string): Subject ID to be included in file structure
#         task_name (string): Name of task to be included in file structure
#         exp_type (string): Type of experiment, either 'pre', 'post', or 'exp'
#         save_dir (string): Name of base directly to save experiment files
        
#     '''
    
#     #create directory structure to find capture files
#     read_folder = os.path.join(read_dir, subject_name, task_name, exp_type)
#     scene_cam_read_folder = os.path.join(read_folder,'scene_camera')
#     eye_cam_read_folder = os.path.join(read_folder,'eye_camera')
#     imu_read_folder = os.path.join(read_folder, 'imu')
    
#     #create directory structure to save analyzed files
#     save_folder = os.path.join(save_dir, subject_name, task_name, exp_type)
#     scene_cam_save_folder = os.path.join(save_folder,'scene_camera')
#     eye_cam_save_folder = os.path.join(save_folder,'eye_camera')
#     imu_save_folder = os.path.join(save_folder, 'imu')
    
#     #run sceme camera analysis
#     print('Running Frame Analysis...')
#     run_ximea_analysis(scene_cam_read_folder, scene_cam_save_folder,
#                        timestamp_stats=run_timestamp_stats,
#                        convert_ims=run_convert_ims)
    
#     #run eye tracker analysis
#     #run IMU analysis
    
#     print("Finished Anaysis!")
    

    
# def count_missed_frames(timestamp_file, cam_name):
#     with open(timestamp_file, 'r') as f:
#         ts_table=list(zip(line.strip().split('\t') for line in f))
    
#     ts_table = np.squeeze(np.array(ts_table[1:]).astype('float'))
#     total_frames = ts_table.shape[0]
#     nf = ts_table[:,1]
#     df = nf[1:] - nf[:-1] - 1
#     df = np.sum(df)
#     dfp = df / total_frames * 100
    
#     print(f'{cam_name} missed frames total: {df} / {total_frames} = {dfp:0.2f}%')
    
#     return(dfp)
    

# def plot_camera_timing(timestamp_file, figwrite_file, cam_name):
#     with open(timestamp_file, 'r') as f:
#         ts_table=list(zip(line.strip().split('\t') for line in f))
    
#     ts_table = np.squeeze(np.array(ts_table[1:]).astype('float'))
#     ts = ts_table[:,2]
#     dt = ts[1:] - ts[:-1]

#     med_dt = np.median(dt)
#     med_fr = 1/np.median(dt)
#     mean_dt = np.mean(dt)
#     mean_fr = 1/np.mean(dt)

#     plt.plot(dt)
#     plt.axhline(med_dt,label=f'median: {med_dt:.5f} = {med_fr:.2f}fps',c='black')
#     plt.axhline(0.005,label='200fps',c='red')
#     plt.axhline(mean_dt,label=f'mean: {mean_dt:.5f} = {mean_fr:.2f}fps',c='green')
#     plt.plot(dt, label='sample', c='blue')
#     plt.ylabel('dt')
#     plt.legend()
#     plt.title(f'{cam_name} Camera Timing')
#     plt.savefig(figwrite_file)
#     plt.show()
    
# def plot_camera_dframe(timestamp_file, figwrite_file, cam_name):
#     with open(timestamp_file, 'r') as f:
#         ts_table=list(zip(line.strip().split('\t') for line in f))
    
#     ts_table = np.squeeze(np.array(ts_table[1:]).astype('float'))
#     ts = ts_table[:,1]
#     dt = ts[1:] - ts[:-1]

#     med_dt = np.median(dt)
#     mean_dt = np.mean(dt)
    

#     plt.plot(dt)
#     plt.axhline(1,label='All Frames Kept',c='red')
#     plt.axhline(med_dt,label=f'median: {med_dt:.5f}',c='black')
#     plt.axhline(mean_dt,label=f'mean: {mean_dt:.5f}',c='green')
#     plt.plot(dt, label='sample', c='blue')
#     plt.ylabel('dframe')
#     plt.legend()
#     plt.title(f'{cam_name} Camera Dframes')
#     plt.savefig(figwrite_file)
#     plt.show()
    
# def ximea_timestamp_to_framenum(timestamp_file, timestamp):
#     '''
#     Given a unix timestamp, what is the closest frame from a ximea camera recording?
#     Params:
#         timestamp_file (str): path to a timestamp file for this camera
#         timestamp (float): timestamp desired.
#     Returns:
#         i (int): frame number of collection *NOT NFRAME CAMERA COUNTER*) closest to timestamp
#         true_timestamp (float): What is the real timestamp of this frame?
#     '''
#     with open(timestamp_file, 'r') as f:
#         ts_table=list(zip(line.strip().split('\t') for line in f))
    
#     ts_table = np.squeeze(np.array(ts_table[1:]).astype('float'))
#     closest_idx = np.argmin(np.abs(ts_table[:,3]-timestamp))
#     i = int(ts_table[closest_idx,0])
#     true_timestamp = ts_table[closest_idx,3]
#     return(i, true_timestamp)
    
# def ximea_framenum_to_timestamp(timestamp_file, framenum):
#     '''
#     Given a unix timestamp, what is the closest frame from a ximea camera recording?
#     Params:
#         timestamp_file (str): path to a timestamp file for this camera
#         framenum (int): framenum desired ***THIS IS NOT NFRAME CAMERA COUNTER but counter with respect to collection******.
#     Returns:
#         ts (float): timestamp of framenum frame
#     '''
    
#     with open(timestamp_file, 'r') as f:
#         ts_table=list(zip(line.strip().split('\t') for line in f))
    
#     ts_table = np.squeeze(np.array(ts_table[1:]).astype('float'))
#     ts = np.float(ts_table[np.where(ts_table[:,0]==framenum),3])
#     return(ts)

# def ximea_get_frame(frame_number, save_batchsize, cam_name, cam_save_folder, img_dims=(1544,2064), normalize=True):
#     '''
#     Get the filename and offset of a given frame number from the camera.
#     Params:
#         frame_number (int): number of frame desired
#         save_bathsize (int): what was the batchsize during collection?
#         cam_name (str): what is the name of the caera? OD/OS/CY
#         cam_save_folder (str): what is the name of the folder?
#         img_dims (int, int): dimensions of frame reading in.
#     Returns:
#         frame (2d numpy array): 2d array of frame from saved file
#     '''
    
#     file_start = int(np.floor(frame_number/save_batchsize)*save_batchsize)
#     file_end = file_start + save_batchsize - 1
#     frame_offset = frame_number%file_start if file_start>0 else frame_number
#     file_name = f'frames_{file_start}_{file_end}.bin'
#     file_path = os.path.join(cam_save_folder, cam_name, file_name)
    
#     frame = bin_to_im(file_path, frame_offset, img_dims)
    
#     if normalize:
#         frame = frame/75
#         frame[frame > 1] = 1
        
#     return(frame)

# # def bin_to_im(binfile, nframe, dims=(1544,2064)):
# #     '''
# #     convert a single image from 8-bit raw bytes to png image.
# #     Input:
# #         binfile (str): path to binary file
# #         dims (2ple int): What are the dimensions of the iamge?
# #         nframe (int): Which frame number do we want within image?
# #         '''
# #     a = []
# #     # for uint8
# #     nbytes = np.prod(dims)
# #     startbyte = nframe*nbytes
# #     with open(binfile, 'rb') as fn:
# #         fn.seek(startbyte)
# #         bs = fn.read(1)
# #         for i in range(nbytes):
# #             bs = fn.read(1)
# #             bs = int.from_bytes(bs,'big')
# #             a.append(bs)
            
# #     a = np.array(a)
# #     im = a.reshape(dims)
# #     imc = cv2.cvtColor(np.uint8(im), cv2.COLOR_BayerGR2RGB)
    
# #     return(imc)

# # def ximea_get_frame_sequence(frame_numbers, save_batchsize, cam_name, cam_save_folder, img_dims=(1544,2064), normalize=True):
# #     '''
# #     Get the filename and offset of a given sequence of frame numbers from the camera.
# #     Params:
# #         frame_numbers (arrray of ints): frame numbers to grab
# #         save_bathsize (int): what was the batchsize during collection?
# #         cam_name (str): what is the name of the caera? OD/OS/CY
# #         cam_save_folder (str): what is the name of the folder?
# #         img_dims (int, int): dimensions of frame reading in.
# #     Returns:
# #         frame (2d numpy array): 2d array of frame from saved file
# #     '''
    
# #     file_start = int(np.floor(frame_number/save_batchsize)*save_batchsize)
# #     file_end = file_start + save_batchsize - 1
# #     frame_offset = frame_number%file_start if file_start>0 else frame_number
# #     file_name_start = f'frames_{file_start}_{file_end}.bin'
# #     file_path = os.path.join(cam_save_folder, cam_name, file_name)
    
# #     frame_sequence = np.zeros((len(frame_numbers), *img_dims, 3))
    
# #     for i in range(len(frame_numbers)):
# #         fnum = frame_numbers[i]
        
        
# #         frame = []
# #         nbytes = np.prod(dims)
# #         startbyte = nframe*nbytes
# #         with open(binfile, 'rb') as fn:
# #             for i in range(nbytes:
# #                 bs = fn.read(1)
# #                 frame[i] = int.from_bytes(bs,'big')
        
    
    
    
# #     frame = bin_to_im(file_path, frame_offset, img_dims)
    
# #     if normalize:
# #         frame = frame/75
# #         frame[frame > 1] = 1
        
# #     return(frame)



# def pupil_framenum_to_timestamp(timestamp_file, framenum):
#     '''
#     Given a unix timestamp, what is the closest frame from a pupil camera recording?
#     Params:
#         timestamp_file (str): path to a timestamp file for this camera
#         framenum (int): framenum desired
#     Returns:
#         ts (float): timestamp of framenum frame
#     '''
#     timestamp_list = np.load(timestamp_file)
#     ts = timestamp_list[framenum]
#     return(ts)

# def pupil_timestamp_to_framenum(timestamp_file, timestamp):
#     '''
#     Given a unix timestamp, what is the closest frame from a pupil camera recording?
#     Params:
#         timestamp_file (str): path to a timestamp file for this camera
#         timestamp (float): timestamp desired.
#     Returns:
#         i (int): frame number of collection *NOT NFRAME CAMERA COUNTER*) closest to timestamp
#         true_timestamp (float): What is the real timestamp of this frame?
#     '''
#     timestamp_list = np.load(timestamp_file)
#     i = np.argmin(np.abs(timestamp_list-timestamp))
#     true_timestamp = timestamp_list[i]

#     return(i, true_timestamp)

# def pupl_get_framemeans(video_file, frame_start, nframes):
#     '''
#     Caclualte the framemeans for a set of frames in sucession
#     '''
#     vidcap = cv2.VideoCapture(video_file)
#     means = []
#     for i in range(frame_start):
#         success, frame = vidcap.read()
#     for i in range(frame_start, frame_start+nframes):
#         success, frame = vidcap.read()
#         means.append(np.mean(frame))
    
#     return(means)

# def pupil_get_frame(video_file, framenum, normalize=False):
#     '''
#     Grab the Pupil Cam Frame at a given framenum
#     Params:
#         video_file (str): path to a video file for this camera
#         framenum (int): framenumber desired.
#         normalize (bool): normalize the frame?
#     Returns:
#         frame (2d array): image from pupil camera
#     '''
#     vidcap = cv2.VideoCapture(video_file)
#     for i in range(framenum):
#         success, frame = vidcap.read()
#     if(normalize):
#         frame = 255*(frame/np.max(image))
#     return(frame)

# def convert_ximea_time_to_unix_time(timestamp_file, sync_file):
#     '''
#     Convert the ximea camera times to unix timestamps
#     Params:
#         timestamp_file (str): path to .csv file with ximea timesstamps
#         sync_file (str): path to .csv file with sync information
#     Returns:
#         unix_timestamp_array (2d np array): Unix Timestamps Inferred
#     '''
#     with open(timestamp_file, 'r') as f:
#         ts_table=list(zip(line.strip().split('\t') for line in f))
#     ts_table = np.squeeze(np.array(ts_table[1:]).astype(np.double))

#     with open(sync_file, 'r') as f:
#         sync_table=list(zip(line.strip().split('\t') for line in f))
        
#     (unix_pre, cam_pre) = np.array(sync_table[1][0][1:]).astype(np.double)
#     (unix_post, cam_post) = np.array(sync_table[2][0][1:]).astype(np.double)

#     #how far off have the computer and camera timestamps drifted?
#     t_elapsed_unix = np.double(unix_post) - np.double(unix_pre)
#     t_elapsed_cam = np.double(cam_post) - np.double(cam_pre)
#     drift = np.abs(t_elapsed_unix - t_elapsed_cam)
#     print(f'Time Elapsed: {t_elapsed_unix} seconds')
#     print(f'Time Drift pre to post: {drift} seconds')
    
#     # We assume here that time.time() in Linux's 0.001s precision is better than camera's.
#     # Convert Camera timestamps to Unix timestamps.
#     #first convert to [0,1]
#     t_cam_converted = (ts_table[:,2] - cam_pre) / (cam_post - cam_pre)
#     #then convert to wall time
#     t_cam_converted = (t_cam_converted * (unix_post - unix_pre)) + unix_pre
    
#     #assume time in camera is linear, and just change offset at pre.
#     #t_cam_converted = ts_table[:,2] - ts_table[0,2] + unix_pre
    
#     print(f'Start at {t_cam_converted[0]}, end at {t_cam_converted[-1]}')
    
#     #add unix time to data
#     t_cam_converted = np.append(ts_table, np.expand_dims(t_cam_converted,1),axis=1)
    
#     return(t_cam_converted)

# def convert_bin_pngs(filename, first_fnum, save_batchsize, save_folder,  dims=(1544,2064)):
#     '''
#     Take a file saved in .bin format from a ximea camera, and convert it to png images.
#     Parameters:
#         filename (str): file to be converted
#         save_folder (str): folder to save png files
#         im_shape (2pule ints): shape of image
#         img_format (str): Image format files are saved
#     Returns:
#         None
#     '''
#     nbytes = np.prod(dims)
    
#     with open(filename, 'rb') as fn:
#         bs = fn.read(1)
#         for i in range(first_fnum, first_fnum+save_batchsize):
#             save_filepath = os.path.join(save_folder, f'frame_{i}.png')
#             binary_img = []
#             for b in range(nbytes):
#                 binary_img.append(int.from_bytes(fn.read(1),'big'))
#             binary_img = np.array(binary_img)
#             cimage = cv2.cvtColor(np.uint8(binary_img.reshape(dims)),cv2.COLOR_BayerGR2RGB)
#             cv2.imwrite(save_filepath, cimage)



# def convert_trial_directory(camera_dir, camera, save_batchsize, analysis_folder):
#     frame_start = 0
#     bin_file = os.path.join(camera_dir,camera,f'frames_{frame_start}_{frame_start+save_batchsize-1}.bin')
#     cam_folder = os.path.join(analysis_folder,'pngs',camera)
#     try:
#         os.makedirs(cam_folder)
#     except:
#         print('already made cam folder!')

#     print(f'Converting bin to png for folder {os.path.join(camera_dir,camera)}')
#     print(f'Each * is {save_batchsize} frames...')
#     while(os.path.isfile(bin_file)):
#         print('*')
#         convert_bin_pngs(bin_file, frame_start, save_batchsize, cam_folder, dims=(1544,2064))
#         frame_start += save_batchsize
#         bin_file = os.path.join(camera_dir,camera,f'frames_{frame_start}_{frame_start+save_batchsize-1}.bin')
#     print('Done!')