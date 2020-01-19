clear; clc; clf;

im = imread('1.png');
% im = rgb2lab(im);
im = imresize(im,[1080,1920]);

patch = im(1:200,1:200,:);
patchVar = std2(patch)^2;
DoS = 2*patchVar;

spat_sigma = 8;
ims = imbilatfilt(im,DoS,spat_sigma);
% imshow(ims);

imd = double(ims);
X = reshape(imd,size(imd,1)*size(imd,2),3);
coeff = pca(X);
imtransformed = X*coeff;

pc1 = reshape(imtransformed(:,1),size(imd,1),size(imd,2));
pc2 = reshape(imtransformed(:,2),size(imd,1),size(imd,2));
pc3 = reshape(imtransformed(:,3),size(imd,1),size(imd,2));

pc1 = abs(pc1);
pc2 = abs(pc2);
pc3 = abs(pc3);

