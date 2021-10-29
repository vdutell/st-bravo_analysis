function [Pzero, Azero, Ezero] = findZeroPupil(pupil2gaze,P1)

Pzero = fminsearch(@(x) error(x,pupil2gaze),P1);

[Azero, Ezero] = apply_pupil2gaze(pupil2gaze,Pzero,'affine');

end

function E = error(P,pupil2gaze)

[A, E] = apply_pupil2gaze(pupil2gaze,P,'affine');
E = norm([A E]);

end