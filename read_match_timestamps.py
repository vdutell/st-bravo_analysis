import os, stat
import numpy as np
import utils.run_analysis as ana

def read_match_timestamps(base_dir, target_fps, subject, trial, num_cameras=1):
    '''
    Readin timestamps from all devices and match them to a common timestamp, repeating frames if needed.
    
    NOTE: You may need to fix the permissions first to run this 
    
    NOTE2: This works for data where the LAST annotation is for starting the task. Future datasets where this is NOT the case (ie with post calibrations) will fail
    
    NOTE3: This only works with one camera FOR NOW!
    
    Params:
        base_dir (str): directory where we work
        target_fps (int): The sample rate in frames per second that our devices are targeted to
        subject (str): name of subject
        trial (str): name of trial
        num_cameras (int): how many cameras did we collect from? (1=cy, 2=cy,os, 3=cy,od,os)
    Returns:
        common_timeline_table (n by timestamps array): Table matching frame numbers from devices to our common timeline timestamps
    '''
    
    cameras = ['cy','od','os']
    cameras = cameras[:num_cameras]
    #TODO: IMPLEMENT FOR MULTIPLE CAMERAS
    cam = cameras[0]
    
    data_dir = os.path.join(base_dir, 'raw_data')
    output_dir = os.path.join(base_dir, 'analysis')
    
    trial_directory = os.path.join(data_dir, subject, trial, 'pre')

    #ximea (scene cameras)
    ximea_timestamp_file = os.path.join(trial_directory, 'scene_camera', f'timestamps_{cam}.tsv')
    ximea_timesync_file = os.path.join(trial_directory, 'scene_camera', f'timestamp_camsync_{cam}.tsv')

    #pupil (eye cameras)
    pupil_timestamp_file = os.path.join(trial_directory, 'eye_camera','000', 'pupil_timestamps.npy')
    pupil_annotations_file = os.path.join(trial_directory, 'eye_camera','000', 'annotation_timestamps.npy')

    analysis_folder = os.path.join(output_dir, subject, trial,'')
    try:
        os.makedirs(analysis_folder)
    except:
        print(f'Folder {analysis_folder} Already Made!')
        
    ximea_timestamps = ana.convert_ximea_time_to_unix_time(ximea_timestamp_file, ximea_timesync_file)
    ximea_timestamp_converted_path = os.path.join(analysis_folder,f'timestamps_converted_{cam}.tsv')
    np.savetxt(ximea_timestamp_converted_path, ximea_timestamps, fmt='%10.5f', delimiter='\t')
    
    pupil_num='000'

    pupil_ts = np.load(pupil_timestamp_file)[::2]
    pupil_annotations = np.load(pupil_annotations_file)

    #ximea timestamps
    with open(ximea_timestamp_converted_path, 'r') as f:
        timestamps_ximea = list(zip(line.strip().split('\t') for line in f))
        timestamps_ximea = np.squeeze(np.array(timestamps_ximea[1:]).astype('float'))
        timestamps_ximea = timestamps_ximea[:,-1]

    ## SEE NOTE 2
    if(trial == 'cell_phone_1' and subject=='jf'):
        start_task_time = pupil_annotations[-2]
        end_task_time = pupil_annotations[-1]
    else:
        start_task_time = pupil_annotations[-1]
        end_task_time = timestamps_ximea[-1]
    print(f'Task Lasted: {end_task_time-start_task_time} seconds.')
  
    
    start_timestamp = np.max((timestamps_ximea[0], pupil_ts[0]))
    end_timestamp = np.min((timestamps_ximea[-1], pupil_ts[-1]))
    common_timeline = np.arange(start_timestamp, end_timestamp, 1./target_fps)

    ximea_common_timeline_match = np.zeros_like(common_timeline)
    pupil_common_timeline_match = np.zeros_like(common_timeline)
    during_task = np.zeros_like(common_timeline)
    during_calibration = np.zeros_like(common_timeline)

    for i, t in enumerate(common_timeline):
        ximea_common_timeline_match[i] = np.argmin(np.abs(timestamps_ximea - t))
        pupil_common_timeline_match[i] = np.argmin(np.abs(pupil_ts - t))
        if((t > start_task_time) and (t < end_task_time)):
            during_task[i] = 1
            
    common_timeline_table = np.array((common_timeline, ximea_common_timeline_match, pupil_common_timeline_match, during_task, during_calibration)).T
    common_timeline_table_colnames = 'common_timeline\tximea_frame\tpupil_frame\tduring_task\tduring_calibration'
    common_timeline_file_human = os.path.join(analysis_folder,'common_timeline.tsv')
    np.savetxt(common_timeline_file_human, common_timeline_table, delimiter='\t', header=common_timeline_table_colnames)
    common_timeline_file = os.path.join(analysis_folder,'common_timeline.npy')
    np.save(common_timeline_file, common_timeline_table)
    return(common_timeline_table)