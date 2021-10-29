function TRANSF = pupil2gaze_affine(TARGET,AZ,EL,type)

if nargin < 4
    type = 'poly22';
end

if strcmp(type,'affine')
    fixedPoints = [AZ,EL];
    movingPoints = TARGET;

    tform = fitgeotrans(fixedPoints,movingPoints, 'polynomial',2);

    [TRANSF.AZ, TRANSF.EL] = transformPointsInverse(tform, movingPoints(:,1), movingPoints(:,2));

    TRANSF.gof.RMSE = nanmean(sqrt((TRANSF.AZ-AZ).^2 + (TRANSF.EL-EL).^2));
    TRANSF.gof.sse = nansum((TRANSF.AZ-AZ).^2 + (TRANSF.EL-EL).^2);
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
