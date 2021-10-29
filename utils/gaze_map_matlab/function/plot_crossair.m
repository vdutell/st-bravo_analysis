function IMAGE = plot_crossair(IMAGE,Xpos,Ypos)

[SY, SX, C] = size(IMAGE);
R = ceil(SY/70);
W = 2;

%% CROSS
IMAGE((-R:R) + Ypos,(-W:W) + Xpos, 1) = 255;
IMAGE((-R:R) + Ypos,(-W:W) + Xpos, 2:3) = 0;

IMAGE((-W:W) + Ypos,(-R:R) + Xpos, 1) = 255;
IMAGE((-W:W) + Ypos,(-R:R) + Xpos, 2:3) = 0;

%% CIRCLE
[XC YC] = meshgrid((1:SY) - Xpos,(1:SX) - Ypos);
D = sqrt(XC.^2 + YC.^2);

mask = D < (R+W) & D > (R-W);

for i = 1:C
    TMP = IMAGE(:,:,i);
    
%     IND = find(mask(:)>0);
%     [R,C] = ind2sub(size(TMP),IND);

    for r = 1:SX
        for c = 1:SY
            if mask(r,c)
                IMAGE(r,c,1) = 255;
                IMAGE(r,c,2) = 0;
                IMAGE(r,c,3) = 0;
            end
        end
    end

end
