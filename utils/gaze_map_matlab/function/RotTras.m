function OUT = RotTras(IN,R,T,direction)
%% function OUT = RotTras(IN,R,T,direction)
% IN = 3D point or vector of 3D points
% R = rotational matrix (3x3)
% T = translational matrix (3x1)
% direction = forth or back rototranslation
% OUT = rototranslated 3D point or vector of 3D points


if strcmp(direction,'back')
    
    OUT = IN*R' + repmat(T,[size(IN,1) 1]);
else
    
    OUT = (IN - repmat(T,[size(IN,1) 1]))*R;
end