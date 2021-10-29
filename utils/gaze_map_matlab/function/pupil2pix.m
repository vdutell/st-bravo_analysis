function TRANSF = pupil2pix(TARGET_POINTS,TARGET_PUPIL)

% Set up fittype and options.
ft = fittype( 'poly22' );
opts = fitoptions( 'Method', 'LinearLeastSquares' );
opts.Normalize = 'on';
opts.Robust = 'LAR';

% [xData, yData, zData] = prepareSurfaceData(LE.PUPIL.TARGET_X_SELECT,LE.PUPIL.TARGET_Y_SELECT,AZ);
[TRANSF.AZ, TRANSF.gofAZ] = fit([TARGET_POINTS], TARGET_PUPIL(:,1), ft, opts );

% [xData, yData, zData] = prepareSurfaceData(LE.PUPIL.TARGET_X_SELECT,LE.PUPIL.TARGET_X_SELECT,EL);
[TRANSF.EL, TRANSF.gofEL] = fit([TARGET_POINTS], TARGET_PUPIL(:,2), ft, opts );
