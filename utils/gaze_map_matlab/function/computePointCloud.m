%% COMPUTE POINT CLOUD FROM DPETH AND CAMERA PARAMETERS
function PC = computePointCloud(depth,CAMERA,TH_NEAR,TH_FAR)

depth = double(depth);

if nargin<4
    TH_FAR = nanmax(depth(:));
end

if nargin<3
    TH_NEAR = nanmin(depth(:));
end

PC.Z = depth;

PC.Z(PC.Z>TH_FAR | PC.Z<TH_NEAR) = nan;
PC.X = (CAMERA.COLOR_SENSOR.ppx - CAMERA.COLOR_SENSOR.MESH.pix.X) .* PC.Z ./ CAMERA.COLOR_SENSOR.fx;
PC.Y = (CAMERA.COLOR_SENSOR.ppy - CAMERA.COLOR_SENSOR.MESH.pix.Y) .* PC.Z ./ CAMERA.COLOR_SENSOR.fy;

