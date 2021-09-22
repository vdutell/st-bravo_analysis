import pyrealsense2 as rs2
import rospy
import numpy as np
import rosbag
import os
import copy
from scipy.spatial.transform import Rotation
import cv2
np.set_printoptions(suppress=True)
def create_aligned_depth_files(recording_folder, output_folder):
    '''
    Run offline alignment of depth stream to both the world camera and the ximea camera coordines
    '''
    
    depth_dims = (848, 480)
    depth_fps = 60
    rgb_dims = (1280, 720, 3)
    #rgb_dims = (1920, 1080, 3)
    #rgb_fps = 30
    ximea_dims = (2064, 1544, 3)
    #ximea_fps = 200
    #instead of using true ximea and rgb fps, upsample and register just at depth framerate
    ximea_fps = depth_fps
    rgb_fps = depth_fps
    
    #intrinsics & extrinsics for ximea camera
    ximea_distortion = [1.81742550e+03, 0.00000000e+00, 1.04625244e+03, 0.00000000e+00, 1.82025698e+03, 7.55093896e+02, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]

    rgb_to_ximea_rotation = [[9.996304422422012115e-01, -1.885864076914232912e-02, -1.957883068525424763e-02],
[1.866599409722233260e-02, 9.997760038579057706e-01, -9.976110177782640878e-03],
[1.976258098083020359e-02, 9.606965090872355076e-03, 9.997585441568973552e-01]]
    rgb_to_ximea_translation = 1e-3 * np.array((2.853691061569304566e+01, 9.189543619411789932e-02, 2.261175317542426733e+00))

    #file and folder paths
    depth_timestamps = list(np.loadtxt(os.path.join(recording_folder,'depth','timestamps.csv')))
    depth_frames = [np.load(os.path.join(recording_folder,'depth',f'depth_frame_{str(f).zfill(8)}.npy')) for f in range(len(depth_timestamps))]
    
    bag_in_path = os.path.join('/home/vasha/Desktop/backpack_computer_aquisition/depth_align_bag/','sample2.bag')
    bag_out_rgb_path = os.path.join(output_folder,'depth_rgb.bag')
    bag_out_ximea_path = os.path.join(output_folder,'depth_ximea.bag')
        
    aligned_rgb_depth_folder = os.path.join(output_folder,'rgb_aligned_depth')
    os.makedirs(aligned_rgb_depth_folder, exist_ok=True)
    aligned_ximea_depth_folder = os.path.join(output_folder,'ximea_aligned_depth')
    os.makedirs(aligned_ximea_depth_folder, exist_ok=True)
    
    # .bag file for depth info
    try:
        bag_in = rosbag.Bag(bag_in_path)
        bag_out_rgb = rosbag.Bag(bag_out_rgb_path, 'w')
        #keep messages with metadata about RGB and depth devices
        for topic, msg, t in bag_in.read_messages():

            #if topic is about gyro or accelerirometer (sensor 2), remove it in our new file
            if 'sensor_2' in str(topic):
                pass
            
            #don't need image or depth metadata
            elif 'Depth_0/image/metadata' in str(topic):
                depth_metadata_msg = msg
                if('Exposure Roi' in str(msg.key)):
                    bag_out_rgb.write(topic, msg, t)
            elif 'Color_0/image/metadata' in str(topic):
                color_metadata_msg = msg
            elif 'Depth_0/image/data' in str(topic):
                depth_data_msg = msg
            elif 'Color_0/image/data' in str(topic):
                color_data_msg = msg 

            #if topic is depth stream info, ensure correct frame rate and size
            elif '/device_0/sensor_0/Depth_0/info' == str(topic):
                msg.fps = depth_fps
                bag_out_rgb.write(topic, msg, t)
            elif '/device_0/sensor_0/Depth_0/info/camera_info' == str(topic):
                msg.height = depth_dims[1]
                msg.width = depth_dims[0]
                bag_out_rgb.write(topic, msg, t)

            #if topic is RGB stream info, ensure correct frame rate and size
            elif '/device_0/sensor_1/Color_0/info' == str(topic):
                msg.fps = rgb_fps
                bag_out_rgb.write(topic, msg, t)
            elif '/device_0/sensor_1/Color_0/info/camera_info' == str(topic):
                msg.height = rgb_dims[1]
                msg.width = rgb_dims[0]
                bag_out_rgb.write(topic, msg, t)

            else:
                #keep everything else
                bag_out_rgb.write(topic, msg, t)

            #if topic is RGB extrinsics, save them so we can combine them with ximea to rgb extrinsics later to get ximea to depth exrinsics
            if '/device_0/sensor_1/Color_0/tf/0' == str(topic):
                depth_to_rgb_translation_message = msg.translation
                depth_to_rgb_rotation_q_message = msg.rotation

        #fill up bag with depth & RGB frames
        depth_data_topic = '/device_0/sensor_0/Depth_0/image/data'
        color_data_topic = '/device_0/sensor_1/Color_0/image/data'
        depth_metadata_topic = '/device_0/sensor_0/Depth_0/image/metadata'
        color_metadata_topic = '/device_0/sensor_1/Color_0/image/metadata'
        for i, (t, frame) in enumerate(zip(depth_timestamps, depth_frames)):
            #depth_data_msg = copy.deepcopy(sample_depth_data_msg)
            time = rospy.Time(t)
            depth_msg = copy.deepcopy(depth_data_msg)
            depth_msg.height = depth_dims[1]
            depth_msg.width = depth_dims[0]
            depth_msg.header.stamp = time
            depth_msg.header.seq = i
            depth_msg.data = frame.astype('<h').tobytes()
            bag_out_rgb.write(depth_data_topic, depth_msg, time)
            
            #metadata
            depth_meta_msg = copy.deepcopy(depth_metadata_msg)
            depth_meta_msg.key = 'Frame Counter'
            depth_meta_msg.value = str(i)
            bag_out_rgb.write(depth_metadata_topic, depth_meta_msg, time)
            depth_meta_msg = copy.deepcopy(depth_metadata_msg)
            depth_meta_msg.key = 'Frame Timestamp'
            depth_meta_msg.value = str(t)
            bag_out_rgb.write(depth_metadata_topic, depth_meta_msg, time)         
      
            color_msg = copy.deepcopy(color_data_msg)
            color_msg.height = rgb_dims[1]
            color_msg.width = rgb_dims[0]
            color_msg.header.stamp = time
            color_msg.header.seq = i
            color_msg.data = np.zeros(rgb_dims).tobytes()
            bag_out_rgb.write(color_data_topic, color_msg, time)
            
            #metadata
            color_meta_msg = copy.deepcopy(color_metadata_msg)
            color_meta_msg.key = 'Frame Counter'
            color_meta_msg.value = str(i)
            bag_out_rgb.write(color_metadata_topic, color_meta_msg, time)
            color_meta_msg = copy.deepcopy(color_metadata_msg)
            color_meta_msg.key = 'Frame Timestamp'
            color_meta_msg.value = str(t)
            bag_out_rgb.write(color_metadata_topic, color_meta_msg, time)
            
    except Exception as e:
        print('Exception!')
        print(e)
    finally:
        bag_in.close()
        bag_out_rgb.close()
    
    print('Finished Creating Depth -> RGB Bag File')
    
    ###########
    #Now use realsense pipeline to read in frames from .bag to run align_to
    pipe = rs2.pipeline()
    cfg = rs2.config()
    cfg.enable_device_from_file(os.path.join(output_folder,'depth_rgb.bag'))
    profile = pipe.start(cfg)
    playback = profile.get_device().as_playback().set_real_time(False)
    
    align_to = rs2.stream.color
    align = rs2.align(align_to)
    
    last_framenum = -1
    print(f'processing {len(depth_timestamps)} timestamps...')
    while True:
    #for f in range(len(depth_timestamps)):
        frameset = pipe.wait_for_frames()
        if(frameset is not None):
            #make sure we aren't starting over again
            aligned_frames = align.process(frameset)
            df = aligned_frames.get_depth_frame()
            #df = frameset.get_depth_frame() #not aligned - for debugging
            framenum = df.get_frame_number()
            if framenum < last_framenum:
                break;
            elif framenum == last_framenum:
                continue
            else:
                #ts = frameset.get_timestamp()
                #print(df.get_frame_number(), ts)
                aligned_depth_frame = np.array(df.get_data())
                cv2.imwrite(os.path.join(aligned_rgb_depth_folder,
                                         f'depth_frame_{str(framenum).zfill(8)}.png'),
                            aligned_depth_frame)
                last_framenum = framenum
        else:
            break
    pipe.stop()
    
    print('Finished Writing RGB Aligned Depth Files - Deleting Bag file Now.')
    #os.remove(os.path.join(output_folder,'depth_rgb.bag'))
    
    #convert rgb to depth quaternion to rotation matrix
    depth_to_rgb_rotation = Rotation.from_quat(np.array((depth_to_rgb_rotation_q_message.x,
                                                         depth_to_rgb_rotation_q_message.y,
                                                         depth_to_rgb_rotation_q_message.z,
                                                         depth_to_rgb_rotation_q_message.w))).as_dcm()
    depth_to_rgb_translation = np.array((depth_to_rgb_translation_message.x,
                                         depth_to_rgb_translation_message.y,
                                         depth_to_rgb_translation_message.z))
    #combine rgb to depth with ximea to rgb to get ximea  to depth
    depth_to_ximea_rotation = rgb_to_ximea_rotation @ depth_to_rgb_rotation
    depth_to_ximea_translation = rgb_to_ximea_rotation @ depth_to_rgb_translation + rgb_to_ximea_translation

    depth_to_ximea_rotation_q = Rotation.from_dcm(depth_to_ximea_rotation).as_quat()
    
    print(f'RGB -> Depth extrinsics rotations are: {np.array((depth_to_rgb_rotation_q_message.x,depth_to_rgb_rotation_q_message.y,depth_to_rgb_rotation_q_message.z,depth_to_rgb_rotation_q_message.w))}')
    print(f'RGB -> Depth extrinsics translations are: {depth_to_rgb_translation}')
    print(f'Ximea -> Depth extrinsics rotations are: {depth_to_ximea_rotation_q}')
    print(f'Ximea -> Depth extrinsics translations are: {depth_to_ximea_translation}')
    
    #Write Depth -> Ximea Bag File
    try:
        bag_in = rosbag.Bag(bag_in_path)             
        bag_out_ximea = rosbag.Bag(bag_out_ximea_path, 'w') 
        #keep messages with metadata about RGB and depth devices
        for topic, msg, t in bag_in.read_messages():

            #if topic is about gyro or accelerirometer (sensor 2), remove it in our new file
            if 'sensor_2' in str(topic):
                pass
            #don't need image or depth metadata (will replace them)
            elif 'Depth_0/image/metadata' in str(topic):
                if('Exposure Roi' in str(msg.key)):
                    bag_out_ximea.write(topic, msg, t)
                else:
                    pass
            elif 'Color_0/image/metadata' in str(topic):
                pass 
            #image and depth data we will replace
            elif 'Depth_0/image/data' in str(topic):
                depth_data_msg = msg
            elif 'Color_0/image/data' in str(topic):
                ximea_data_msg = msg
            #if topic is depth stream info, ensure correct frame rate and size
            elif '/device_0/sensor_0/Depth_0/info' == str(topic):
                msg.fps = depth_fps
                bag_out_ximea.write(topic, msg, t)
            elif '/device_0/sensor_0/Depth_0/info/camera_info' == str(topic):
                msg.height = depth_dims[1]
                msg.width = depth_dims[0]
                bag_out_ximea.write(topic, msg, t)

            #if topic is RGB stream info, ensure correct frame rate and size
            elif '/device_0/sensor_1/Color_0/info' == str(topic):
                msg.fps = ximea_fps
                bag_out_ximea.write(topic, msg, t)
            elif '/device_0/sensor_1/Color_0/info/camera_info' == str(topic):
                msg.height = ximea_dims[1]
                msg.width = ximea_dims[0]
                msg.K = ximea_distortion
                bag_out_ximea.write(topic, msg, t)

            #if topic is RGB extrinsics, replace them with ximea to depth exrinsics
            elif '/device_0/sensor_1/Color_0/tf/0' == str(topic):
                msg.rotation.x = depth_to_ximea_rotation_q[0]
                msg.rotation.y = depth_to_ximea_rotation_q[1]
                msg.rotation.z = depth_to_ximea_rotation_q[2]
                msg.rotation.w = depth_to_ximea_rotation_q[3]
                msg.translation.x = depth_to_ximea_translation[0]
                msg.translation.y = depth_to_ximea_translation[1]
                msg.translation.z = depth_to_ximea_translation[2]
            else:
                #keep everything else
                bag_out_ximea.write(topic, msg, t)

        #fill up bag with depth & ximea frames
        depth_data_topic = '/device_0/sensor_0/Depth_0/image/data'
        ximea_data_topic = '/device_0/sensor_1/Color_0/image/data'
        depth_metadata_topic = '/device_0/sensor_0/Depth_0/image/metadata'
        ximea_metadata_topic = '/device_0/sensor_1/Color_0/image/metadata'
        ximea_metadata_msg = color_metadata_msg
        for i, (t, frame) in enumerate(zip(depth_timestamps, depth_frames)):
            time = rospy.Time(t)
            depth_data_msg.height = depth_dims[1]
            depth_data_msg.width = depth_dims[0]
            depth_data_msg.header.stamp = time
            depth_data_msg.header.seq = i
            depth_data_msg.data = frame.tobytes()
            bag_out_ximea.write(depth_data_topic, depth_data_msg, time)
            
            depth_metadata_msg.key = 'Frame Counter'
            depth_metadata_msg.value = str(i)
            bag_out_ximea.write(depth_metadata_topic, depth_metadata_msg, time)
            depth_metadata_msg.key = 'Frame Timestamp'
            depth_metadata_msg.value = str(t)
            bag_out_ximea.write(depth_metadata_topic, depth_metadata_msg, time)        
            
            ximea_data_msg.height = ximea_dims[1]
            ximea_data_msg.width = ximea_dims[0]
            ximea_data_msg.header.stamp = time
            ximea_data_msg.header.seq = i
            ximea_data_msg.data = np.zeros(ximea_dims).tobytes()
            bag_out_ximea.write(ximea_data_topic, ximea_data_msg, time)
            
            ximea_metadata_msg.key = 'Frame Counter'
            ximea_metadata_msg.value = str(i)
            bag_out_ximea.write(ximea_metadata_topic, ximea_metadata_msg, time)
            ximea_metadata_msg.key = 'Frame Timestamp'
            ximea_metadata_msg.value = str(t)
            bag_out_ximea.write(ximea_metadata_topic, ximea_metadata_msg, time)
    except Exception as e:
        print('Exception!')
        print(e)
    finally:
        bag_in.close()
        bag_out_ximea.close()
    print('Finished Creating Depth -> Ximea Bag File')

    ###########
    #Now use realsense pipeline to read in frames from .bag to run align_to
    pipe = rs2.pipeline()
    cfg = rs2.config()
    cfg.enable_device_from_file(os.path.join(output_folder,'depth_ximea.bag'))
    profile = pipe.start(cfg)
    playback = profile.get_device().as_playback().set_real_time(False)
    
    align_to = rs2.stream.color
    align = rs2.align(align_to)
    
    last_framenum = -1
    print(f'processing {len(depth_timestamps)} timestamps...')
    while True:
    #for f in range(len(depth_timestamps)):
        frameset = pipe.wait_for_frames()
        if(frameset is not None):
            #make sure we aren't starting over again
            aligned_frames = align.process(frameset)
            df = aligned_frames.get_depth_frame()
            #df = frameset.get_depth_frame() #not aligned - for debugging
            framenum = df.get_frame_number()
            if framenum < last_framenum:
                break;
            elif framenum == last_framenum:
                continue
            else:
                #ts = frameset.get_timestamp()
                #print(df.get_frame_number(), ts)
                aligned_depth_frame = np.array(df.get_data())
                cv2.imwrite(os.path.join(aligned_ximea_depth_folder,
                                         f'depth_frame_{str(framenum).zfill(8)}.png'),
                            aligned_depth_frame)
                last_framenum = framenum
        else:
            break
    pipe.stop()
    
    print('Finished Writing Ximea Aligned Depth Files - Deleting Bag File Now')
    #os.remove(os.path.join(output_folder,'depth_ximea.bag'))

    print('All Done!')


create_aligned_depth_files(recording_folder='/home/vasha/recordings/2021_04_19/003', output_folder='./depth_align_bag/')
    
