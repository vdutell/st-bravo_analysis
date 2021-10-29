function [INTERSECTION, INT_POINTS, WEIGHTS] = computeBestIntersection(X0,LMN)
% function [INTERSECTION] = computeBestIntersection(X0,LMN)
% compute the best intersection of a set of lines in parametric form
% INPUT
% X0: Nx3 matrix of threedimensional points
% LMN: Nx3 matrix of threedimensional versors
% OUTPUT
% INTERSECTION: point of minimum distance manong all lines


%% LINE DISTANCE BETWEEN PAIRS OF LINES
for r = 1:9
    for c = 1:9
        [P D] = rect_distance(X0(r,:),LMN(r,:),X0(c,:),LMN(c,:));
        INT_POINTS(:,r,c) = P;
%         DIST(r,c) = abs(D);
    end
end
INT_POINTS = permute(INT_POINTS,[2 3 1]);
INT_POINTS = reshape(INT_POINTS,[9*9 3]);

%% DISTANCE BETWEEN PAIRS OF POINTS
for r = 1:length(INT_POINTS)
    for c = 1:length(INT_POINTS)        
%         DIST_P(r,c) = norm(POINT(r,:)-POINT(c,:));
        DIST_P(r,c) = sqrt(nansum((INT_POINTS(r,:)-INT_POINTS(c,:)).^2));
%         DIST_P(r,c)
%         aa=0
    end
end

%% THRESHOLD AND NORMALIZATION
DIST_P = nansum(DIST_P); DIST_P(DIST_P==0) = nan;
DIST_P(abs(DIST_P)>prctile(DIST_P(:),30)) = nan;

WEIGHTS = 1./(DIST_P + 0.01);
WEIGHTS = WEIGHTS ./ nansum(WEIGHTS(:));
 
%% WEIGHTED MEAN
INTERSECTION = squeeze(nansum(INT_POINTS.*repmat(WEIGHTS(:),[1 3])));

% INTERSECTION = nanmedian(POINT);
% DIST = nanmedian(DIST(:));