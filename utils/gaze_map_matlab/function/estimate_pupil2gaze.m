function TRANSF = estimate_pupil2gaze(TARGET,AZ,EL,type)

if nargin < 4
    type = 'poly22';
end

mask_nan = isnan(TARGET(:,1)) | isnan(TARGET(:,2)) | isnan(AZ) | isnan(EL);
TARGET(mask_nan,:) = [];
AZ(mask_nan) = [];
EL(mask_nan) = [];

TRANSF.type = type;

if strcmp(type,'affine')
    fixedPoints = [AZ,EL];
    movingPoints = TARGET;

    TRANSF.tform = fitgeotrans(fixedPoints,movingPoints, 'polynomial',2);

    [TRANSF_AZ, TRANSF_EL] = transformPointsInverse(TRANSF.tform, movingPoints(:,1), movingPoints(:,2));

    TRANSF.gof.RMSE = nanmean(sqrt((TRANSF_AZ-AZ).^2 + (TRANSF_EL-EL).^2));
    TRANSF.gof.sse = nansum((TRANSF_AZ-AZ).^2 + (TRANSF_EL-EL).^2);

end


if strcmp(type,'poly22')
    % Set up fittype and options.
    ft = fittype( 'poly22' );
    opts = fitoptions( 'Method', 'LinearLeastSquares' );
    opts.Normalize = 'on';
    opts.Robust = 'LAR';

    % [xData, yData, zData] = prepareSurfaceData(LE.PUPIL.TARGET_X_SELECT,LE.PUPIL.TARGET_Y_SELECT,AZ);
    [TRANSF.AZ, TRANSF.gofAZ] = fit([TARGET], AZ, ft, opts );

    % [xData, yData, zData] = prepareSurfaceData(LE.PUPIL.TARGET_X_SELECT,LE.PUPIL.TARGET_X_SELECT,EL);
    [TRANSF.EL, TRANSF.gofEL] = fit([TARGET], EL, ft, opts );
end
