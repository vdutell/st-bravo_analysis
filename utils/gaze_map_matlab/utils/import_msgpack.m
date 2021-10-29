function import_msgpack(folder)

%% export msgpack to txt
file = '/offline_data/reference_locations';
if ~exist(fullfile(folder,file,'.txt'))
    system(['python ./utils/read_msgpack.py ' folder file]);
end

%% import txt
DATA = importdata([folder file '.txt'],'r');
DATA = DATA{1};
DATA(end-1:end) = [];
DATA(1:26) = [];

%% parse txt
STOP = false;
count = 1;
while ~STOP
    try
        DATA(1:2) = [];
        [token,DATA] = strtok(DATA,', ');        
        DATA_MAT(count,2) = str2num(token);
        
        [token,DATA] = strtok(DATA,'],');        
        DATA_MAT(count,3) = str2num(token);
        
        DATA(1) = [];
        [token,DATA] = strtok(DATA,', ');        
        DATA_MAT(count,4) = str2num(token);
        
        [token,DATA] = strtok(DATA,'], ');        
        DATA_MAT(count,1) = str2num(token);
        
        DATA(1:3) = [];
        count = count + 1;
    catch 
        STOP = true;
    end
end

%% export to marker_center.csv
DATA_MAT_EXP(:,1:3) = DATA_MAT(:,1:3);
DATA_MAT_EXP(:,4:5) = nan(size(DATA_MAT,1),2);
DATA_MAT_EXP(:,6) = DATA_MAT(:,4);

if ~exist([folder '\3d_calibration'])
    mkdir([folder '\3d_calibration'])
end
csvwrite([folder '\3d_calibration\marker_center_NEW.csv'],DATA_MAT_EXP)
