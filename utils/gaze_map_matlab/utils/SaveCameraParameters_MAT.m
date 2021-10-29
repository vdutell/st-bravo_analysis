%% SAVE CAMERA PARAMETERS
clear
clc

CAMERA.MODEL = 'D415';
CAMERA.SERIAL = '739112060978';

extrinsics = get_extrinsics_to(this, to)
intrinsics = get_intrinsics(this)

%% COLOR
COLOR_SENSOR.width = 1920; COLOR_SENSOR.height = 1080;
COLOR_SENSOR.ppx = 958.083; COLOR_SENSOR.ppy = 530.833;
COLOR_SENSOR.fx = 1372.89; COLOR_SENSOR.fy = 1374.39;
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

DEPTH_SENSOR.width = 848; DEPTH_SENSOR.height = 640;
DEPTH_SENSOR.ppx = 424.485; DEPTH_SENSOR.ppy = 233.825;
DEPTH_SENSOR.fx = 630.759; DEPTH_SENSOR.fy = 630.759;
DEPTH_SENSOR.FOV = 2.*[atand(0.5*DEPTH_SENSOR.width / DEPTH_SENSOR.fx)...
                       atand(0.5*DEPTH_SENSOR.height / DEPTH_SENSOR.fy)]; %deg

[DEPTH_SENSOR.MESH.pix.X DEPTH_SENSOR.MESH.pix.Y] = meshgrid(1:DEPTH_SENSOR.width,1:DEPTH_SENSOR.height);
[DEPTH_SENSOR.MESH.deg.X DEPTH_SENSOR.MESH.deg.Y] = meshgrid(linspace(-DEPTH_SENSOR.FOV(1),DEPTH_SENSOR.FOV(1),DEPTH_SENSOR.width),...
                          linspace(-DEPTH_SENSOR.FOV(2),DEPTH_SENSOR.FOV(2),DEPTH_SENSOR.height));                   

DEPTH_SENSOR.FRUSTUM.Z = ones(1,5);
DEPTH_SENSOR.FRUSTUM.X = ([1 1 DEPTH_SENSOR.width DEPTH_SENSOR.width 1]   - DEPTH_SENSOR.ppx)...
                            .* DEPTH_SENSOR.FRUSTUM.Z ./ DEPTH_SENSOR.fx;
DEPTH_SENSOR.FRUSTUM.Y = ([1 DEPTH_SENSOR.height DEPTH_SENSOR.height 1 1] - DEPTH_SENSOR.ppy)...
                            .* DEPTH_SENSOR.FRUSTUM.Z ./ DEPTH_SENSOR.fy; 

CAMERA.DEPTH_SENSOR = DEPTH_SENSOR;

save(['DATA\CameraIntrinsic_' CAMERA.MODEL '_' CAMERA.SERIAL],'CAMERA')

