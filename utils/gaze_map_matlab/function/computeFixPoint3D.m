function BIN_EYE = computeFixPoint3D(LEFT_EYE_GAZE,RIGHT_EYE_GAZE,CALIB)

PL = computeEULER_to_XYZ(LEFT_EYE_GAZE(:,1), LEFT_EYE_GAZE(:,2));   %% eye  centered coordinates
PL = RotTras(PL,CALIB.LE.R,0,'back');                    %% head-centered coordinates

PR = computeEULER_to_XYZ(RIGHT_EYE_GAZE(:,1), RIGHT_EYE_GAZE(:,2));
PR = RotTras(PR,CALIB.RE.R,0,'back');

for i = 1:size(PL,1)
    [P(i,:) D(i,:)] = rect_distance(CALIB.LE.POS,PL(i,:),CALIB.RE.POS,PR(i,:));
end

% OL = CALIB.LE.POS;
% OR = CALIB.RE.POS;
% OB = 0.5 * (OL + OR);
% for i = 1:size(PL,1)
%     
%     TL = PL(i,:) - OL;
%     TR = PR(i,:) - OR;
%     
%     n = cross(TL,TR);
%     
%     A = norm(TL).^2;
%     B = 2*(TL*OL' - TL*OR');
%     C = 2*TL*TR';
%     D = 2*(TR*OR' - TR*OL');
%     E = norm(TR).^2;
%     F = norm(OL).^2 + norm(OR).^2;
%     
%     tR = (2*A*D + B*C) ./ (C.^2 - 4*A*E);
%     tL = (C*tR - B) ./ (2*A);
%     
%     NPL(i,:) = tL*TL + OL;
%     NPR(i,:) = tR*TR + OR;
%         
% %     figure,hold on
% %     plot3(OL(1),OL(3),OL(2),'ob')
% %     plot3(OR(1),OR(3),OR(2),'or')
% %     
% %     plot3(NPL(1),NPL(3),NPL(2),'ob')
% %     plot3(NPR(1),NPR(3),NPR(2),'or')
%     
%     PB(i,:) = 0.5*(NPL(i,:) + NPR(i,:));
%     TB(i,:) = PB(i,:) - OB;
%     TB(i,:) = TB(i,:) ./ norm(TB(i,:)); 
%     
%     ERROR(i) = norm(NPL(i,:) - NPR(i,:));
%     
% % 	[Intersection_to_all, Intersection_to_line] = computeOptimalIntersection([OL;OR],[TL;TR]);
% 
% end

BIN_EYE.FP = P;
BIN_EYE.FixDist = sqrt(sum(P.^2,2));
% BIN_EYE.FPL = PB;
% BIN_EYE.FPR = PB;
BIN_EYE.ERROR = D;
BIN_EYE.FixPointNUM = size(P,1);
% BIN_EYE.POS = OB;











