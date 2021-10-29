function P = computeEULER_to_XYZ(AZ, EL)
% compute the XYZ of a 3D point P, starting from EULER angles azimuth AZ
% and elevation EL (intended in a coordinate system centered in the eye)
% the computed vectors will have unit norm

for i = 1:length(AZ)
    Y = -sind(EL(i));
    Z = cosd(EL(i)).*cosd(AZ(i));
    X = cosd(EL(i)).*sind(AZ(i));
    
    P(i,:) = [X Y Z];
end




