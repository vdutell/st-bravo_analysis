function [P DIST] = rect_distance(X01,LMN1,X02,LMN2)
% function D = rect_distance(X01,LMN1,X02,LMN2)
% compute the distance between rect 1 and rect 2, expressed in parametric
% equation (X1,LMN1), (X2,LMN2)

N = cross(LMN1,LMN2);
N1 = cross(LMN1,N);
N2 = cross(LMN2,N);

DIST = dot(N,(X01 - X02)) ./ norm(N);

P1 = X01 + dot((X02 - X01),N2) ./ dot(LMN1,N2) .* LMN1;
P2 = X02 + dot((X01 - X02),N1) ./ dot(LMN2,N1) .* LMN2;

P = 0.5 * (P1 + P2);

DIST_CHK = sqrt(sum((P1 - P2).^2));
