function[I2,center]=FindCircle(Io)
close all

I=uint8(Io);
% figure,imshow(I);

% Ostu's rule is used for thresholding you can give your own threshold if
% resuls are not satisfactory.
level = graythresh(I);
BW = im2bw(I,level);
figure,imshow(BW);

%removal of small objects
P=3000;
BW2 = bwareaopen(BW,P);
% figure,imshow(BW2);

BW3 = bwmorph(BW2,'dilate');
% figure,imshow(BW3)

BW4 = bwmorph(BW3,'remove');
% figure, imshow(BW4)

% [I2,center]=Hughcir(BW4);
% figure,imshow(I2)
accum = CircularHough_Grd(BW4, [100 300]);
[~, posTarget]=max2(accum);