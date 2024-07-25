RGB = imread(fullpath(getIPCVpath() + "/images/baboon.png"));
S = rgb2gray(RGB);
Sbin = im2bw(S,0.5);
imshow(Sbin);
