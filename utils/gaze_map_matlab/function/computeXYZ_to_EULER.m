function [R Rinv AZ EL] = computeXYZ_to_EULER(P,O)
% compute the AZIMUTH and ELEVATION of a 3D point P, with respect to an
% origin O, and the rotation matrix R to rotate it to [0 0 norm(P)]

if nargin <2
    O = [0 0 0];
end

for i = 1:size(P,1)
    P(i,:) = P(i,:) - O;
    
    AZ(i) = atand(P(i,1) ./ P(i,3));

    RY(:,:,i) = [cosd(AZ(i)) 0 sind(AZ(i));
          0        1        0;
         -sind(AZ(i)) 0 cosd(AZ(i))];

    CENTRAL_POINT_Y =  P(i,:)*(RY(:,:,i));
    EL(i) = atand(-CENTRAL_POINT_Y(2) ./ CENTRAL_POINT_Y(3));


    RX(:,:,i) = [1        0        0;
          0 cosd(EL(i)) -sind(EL(i));
          0 sind(EL(i)) cosd(EL(i))]; 

    RZ(:,:,i) = eye(3);

    R(:,:,i) = RY(:,:,i)*RX(:,:,i)*RZ(:,:,i);

    Rinv(:,:,i) = inv(R(:,:,i));
end


AZ = AZ';
EL = EL';

% test
% P*(R)