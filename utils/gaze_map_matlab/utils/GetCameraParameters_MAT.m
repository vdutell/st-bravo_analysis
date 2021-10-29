%% SAVE CAMERA PARAMETERS
clear
clc

addpath 'C:\Program Files (x86)\Intel RealSense SDK 2.0\matlab'

% create pipeline
pipe = realsense.pipeline();

% set desired configuration
cfg = realsense.config;
% cfg.enable_stream(realsense.stream.depth,848,480,realsense.format.z16, 30);
% cfg.enable_stream(realsense.stream.color, 1280,720,realsense.format.bgr8, 30);
cfg.enable_stream(realsense.stream.depth,848,480,realsense.format.z16, 30);
% cfg.enable_stream(realsense.stream.color, 640,480,realsense.format.bgr8, 30);
cfg.enable_stream(realsense.stream.color, 960,540,realsense.format.bgr8, 30);

% Start streaming on an arbitrary camera with default settings
pipeline_profile = pipe.start(cfg);

% Get streaming device's name and serial
dev = pipeline_profile.get_device();
camera_name = dev.get_info(realsense.camera_info.name);
serial_number = dev.get_info(realsense.camera_info.serial_number);

tmp = strsplit(flip(camera_name),' ');
CAMERA.MODEL = fliplr(tmp{1});
CAMERA.SERIAL = serial_number;

% get depth intrinsic
depth_stream     = pipeline_profile.get_stream(realsense.stream.depth);
depth_stream     = depth_stream.as('video_stream_profile');      % <--- 'Access violation'
DEPTH_SENSOR     = depth_stream.get_intrinsics();


% get color intrinsic
color_stream     = pipeline_profile.get_stream(realsense.stream.color);
color_stream     = color_stream.as('video_stream_profile');      % <--- 'Access violation'
COLOR_SENSOR     = color_stream.get_intrinsics();

% get depth to color extrinsic
pipe.stop()

% 
% extrinsics = get_extrinsics_to(this, to)
% intrinsics = get_intrinsics(this)
% 
%% COLOR
% % % COLOR_SENSOR.width = 1920; COLOR_SENSOR.height = 1080;
% % % COLOR_SENSOR.ppx = 958.083; COLOR_SENSOR.ppy = 530.833;
% % % COLOR_SENSOR.fx = 1372.89; COLOR_SENSOR.fy = 1374.39;
COLOR_SENSOR.width = double(COLOR_SENSOR.width);
COLOR_SENSOR.height = double(COLOR_SENSOR.height);
COLOR_SENSOR.FOV = 2.*[atand(0.5*COLOR_SENSOR.width / COLOR_SENSOR.fx)...
                       atand(0.5*COLOR_SENSOR.height / COLOR_SENSOR.fy)]; %deg
                   
[COLOR_SENSOR.MESH.pix.X COLOR_SENSOR.MESH.pix.Y] = meshgrid(1:COLOR_SENSOR.width,1:COLOR_SENSOR.height);
[COLOR_SENSOR.MESH.deg.X COLOR_SENSOR.MESH.deg.Y] = meshgrid(linspace(-COLOR_SENSOR.FOV(1),COLOR_SENSOR.FOV(1),COLOR_SENSOR.width),...
                          linspace(-COLOR_SENSOR.FOV(2),COLOR_SENSOR.FOV(2),COLOR_SENSOR.height));                   

COLOR_SENSOR.FRUSTUM.Z = ones(1,5);
COLOR_SENSOR.FRUSTUM.X = ([1 1 COLOR_SENSOR.width COLOR_SENSOR.width 1]   - COLOR_SENSOR.ppx)...
                            .* COLOR_SENSOR.FRUSTUM.Z ./ COLOR_SENSOR.fx;
COLOR_SENSOR.FRUSTUM.Y = ([1 COLOR_SENSOR.height COLOR_SENSOR.height 1 1] - COLOR_SENSOR.ppy)...
                            .* COLOR_SENSOR.FRUSTUM.Z ./ COLOR_SENSOR.fy; 

CAMERA.COLOR_SENSOR = COLOR_SENSOR;

%% DEPTH
% % % DEPTH_SENSOR.width = 848; DEPTH_SENSOR.height = 640;
% % % DEPTH_SENSOR.ppx = 424.485; DEPTH_SENSOR.ppy = 233.825;
% % % DEPTH_SENSOR.fx = 630.759; DEPTH_SENSOR.fy = 630.759;
DEPTH_SENSOR.width = double(DEPTH_SENSOR.width);
DEPTH_SENSOR.height = double(DEPTH_SENSOR.height);
DEPTH_SENSOR.FOV = 2.*[atand(0.5*DEPTH_SENSOR.width / DEPTH_SENSOR.fx)...
                       atand(0.5*DEPTH_SENSOR.height / DEPTH_SENSOR.fy)]; %deg

[DEPTH_SENSOR.MESH.pix.X DEPTH_SENSOR.MESH.pix.Y] = meshgrid(1:DEPTH_SENSOR.width,1:DEPTH_SENSOR.height);
[DEPTH_SENSOR.MESH.deg.X DEPTH_SENSOR.MESH.deg.Y] = meshgrid(linspace(-DEPTH_SENSOR.FOV(1)/2,DEPTH_SENSOR.FOV(1)/2,DEPTH_SENSOR.width),...
                          linspace(-DEPTH_SENSOR.FOV(2)/2,DEPTH_SENSOR.FOV(2)/2,DEPTH_SENSOR.height));                   

DEPTH_SENSOR.FRUSTUM.Z = ones(1,5);
DEPTH_SENSOR.FRUSTUM.X = ([1 1 DEPTH_SENSOR.width DEPTH_SENSOR.width 1]   - DEPTH_SENSOR.ppx)...
                            .* DEPTH_SENSOR.FRUSTUM.Z ./ DEPTH_SENSOR.fx;
DEPTH_SENSOR.FRUSTUM.Y = ([1 DEPTH_SENSOR.height DEPTH_SENSOR.height 1 1] - DEPTH_SENSOR.ppy)...
                            .* DEPTH_SENSOR.FRUSTUM.Z ./ DEPTH_SENSOR.fy; 

CAMERA.DEPTH_SENSOR = DEPTH_SENSOR;

[file,path] = uiputfile(['CameraIntrinsic_' CAMERA.MODEL '_' num2str(COLOR_SENSOR.height) 'p_' CAMERA.SERIAL '.mat'],'Save file name');
save([path file],'CAMERA')

