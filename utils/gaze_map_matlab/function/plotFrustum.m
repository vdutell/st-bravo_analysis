%% PLOT FRUSTUM
function h = plotFrustum(FRUSTUM,FAR_PLANE)

FRUSTUM.X = FRUSTUM.X .* FAR_PLANE;
FRUSTUM.Y = FRUSTUM.Y .* FAR_PLANE;
FRUSTUM.Z = FRUSTUM.Z .* FAR_PLANE;

h{1} = plot3(FRUSTUM.X,FRUSTUM.Z,FRUSTUM.Y,'k');
h{2} = plot3([FRUSTUM.X;zeros(1,5)],[FRUSTUM.Z;zeros(1,5)],[FRUSTUM.Y;zeros(1,5)],'k');

