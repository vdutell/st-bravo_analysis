function PIX_WORLD = computeMonoPixel(PUPIL,CALIB_EYE)


[X_FIT,Y_FIT] = transformPointsForward(CALIB_EYE.pupil2pix,PUPIL.X,PUPIL.Y);

PIX_WORLD = [X_FIT,Y_FIT];

