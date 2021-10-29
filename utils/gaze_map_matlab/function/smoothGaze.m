function GAZEs = smoothGaze(GAZE)

GX = GAZE(:,1);
GY = GAZE(:,2);

GXd = GX - smooth(GX,7,'lowess');
GYd = GY - smooth(GY,7,'lowess');

[HX, binX] = hist(abs(GXd),linspace(0,2,101));
[HY, binY] = hist(abs(GYd),linspace(0,2,101));

[~, upper_TH_X] = triangular_threshold(HX, binX);
[~, upper_TH_Y] = triangular_threshold(HY, binY);

mask = GXd > upper_TH_X | GYd > upper_TH_Y;

GXs = GX; GXs(mask) = nan;
GYs = GY; GYs(mask) = nan;

GXs = smooth(GXs,7,'lowess');
GYs = smooth(GYs,7,'lowess');

GAZEs = [GXs, GYs];
