function [Intersection_to_all, Intersection_to_line] = computeOptimalIntersectionRANSAC(X0,n)
% function [INTERSECTION] = computeOptimalIntersection(X0,LMN)
% compute the optimal intersection of a set of lines in parametric form
% INPUT
% X0: Nx3 matrix of threedimensional points
% LMN: Nx3 matrix of threedimensional versors
% OUTPUT
% Intersection_to_all: point of minimum distance TO all lines
% Intersection_to_line: point of minimum distance ON all lines
X0_orig = X0;
n_orig = n;

SAMPLE_NUM = length(n);

ITER = 1000;
PERC = 10;
RANSAC_SAMPLE_NUM = ceil(SAMPLE_NUM*PERC/100);

ERROR = zeros(SAMPLE_NUM,1);

for i = 1:ITER
    
    [dumb,SELECT] = sort(rand(SAMPLE_NUM,1));
    SELECT = SELECT(1:RANSAC_SAMPLE_NUM);
    
    X0 = X0_orig(SELECT,:);
    n = n_orig(SELECT,:);
    
    
    ux = n(:,1)./(sqrt(sum(n.^2,2)));
    uy = n(:,2)./(sqrt(sum(n.^2,2)));
    uz = n(:,3)./(sqrt(sum(n.^2,2)));
    
    ux = diag(ux);
    uy = diag(uy);
    uz = diag(uz);
    
    u = zeros(3*size(X0,1),size(X0,1));
    u(1:3:(3*size(X0,1)^2)) = -ux(:);
    u(2:3:(3*size(X0,1)^2)) = -uy(:);
    u(3:3:(3*size(X0,1)^2)) = -uz(:);
    
    ux = diag(ux);
    uy = diag(uy);
    uz = diag(uz);
    
    G = [repmat(eye(3),size(X0,1),1) u];
    
    d = zeros(3*size(X0,1),1);
    d(1:3:3*size(X0,1)) = X0(:,1);
    d(2:3:3*size(X0,1)) = X0(:,2);
    d(3:3:3*size(X0,1)) = X0(:,3);
    
    [U,S,V] = svd(G);
    
    s = diag(S);
    mask = logical(s>1e-10);
    
    msvd = (V(:,mask)*inv(S(mask,mask))*(U(:,mask)'))*d;
    
    %Compute the nearest-point TO all the lines
    Intersection_to_all(i,:) = msvd(1:3)';
    
    %Compute the nearest-point ON all the lines
    Intersection_to_line = [X0(:,1)+ux.*msvd(4:end),X0(:,2)+uy.*msvd(4:end),X0(:,3)+uz.*msvd(4:end)];
    
    
    %% COMPUTE ERROR
    
    X1_orig = X0_orig + n_orig;
    
    for nn = 1:SAMPLE_NUM
        D(nn,1) = norm(cross(Intersection_to_all(i,:) - X0_orig(nn,:),Intersection_to_all(i,:) - X1_orig(nn,:)));
    end
    
    ERROR = ERROR + D;
end

SELECT = ERROR < median(ERROR);

X0 = X0_orig(SELECT,:);
n = n_orig(SELECT,:);


ux = n(:,1)./(sqrt(sum(n.^2,2)));
uy = n(:,2)./(sqrt(sum(n.^2,2)));
uz = n(:,3)./(sqrt(sum(n.^2,2)));

ux = diag(ux);
uy = diag(uy);
uz = diag(uz);

u = zeros(3*size(X0,1),size(X0,1));
u(1:3:(3*size(X0,1)^2)) = -ux(:);
u(2:3:(3*size(X0,1)^2)) = -uy(:);
u(3:3:(3*size(X0,1)^2)) = -uz(:);

ux = diag(ux);
uy = diag(uy);
uz = diag(uz);

G = [repmat(eye(3),size(X0,1),1) u];

d = zeros(3*size(X0,1),1);
d(1:3:3*size(X0,1)) = X0(:,1);
d(2:3:3*size(X0,1)) = X0(:,2);
d(3:3:3*size(X0,1)) = X0(:,3);

[U,S,V] = svd(G);

s = diag(S);
mask = logical(s>1e-10);

msvd = (V(:,mask)*inv(S(mask,mask))*(U(:,mask)'))*d;

%Compute the nearest-point TO all the lines
Intersection_to_all = msvd(1:3)';