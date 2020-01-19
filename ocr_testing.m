clear; clc; clf;

im = imread('2.png');
im = imresize(im,[1080,1920]);

patch = im(1:200,1:200,:);
patchVar = std2(patch)^2;
DoS = 2*patchVar;

spat_sigma = 8;
ims = imbilatfilt(im,DoS,spat_sigma);


% poi = ims(300:390,730:875,:);%1.png
poi = ims(650:770,1200:1335,:);%2.png
% poi = im(500:560,1290:1330,:);%3.png

poi = rgb2gray(poi);

poi_corrected = imtophat(poi,strel('disk',15));

marker = imerode(poi_corrected,strel('line',10,0));
poi_clean = imreconstruct(marker,poi_corrected);

poi_bin = imbinarize(poi_clean);
poi_bin_image = 255*[uint8(double(poi_bin))];

imshowpair(poi_clean,poi_bin,'montage');

O = ocr(poi_bin,'CharacterSet','ABCDEFGHIJKLMNOPQRSTUVWXYZ','TextLayout','Block')

[sortedConf,sortedIndex] = sort(O.CharacterConfidences,'descend');
idx_nan_removed = sortedIndex(~isnan(sortedConf));
best_idx = idx_nan_removed(1);

most_confident_letter = num2cell(O.Text(best_idx));
bboxes = O.CharacterBoundingBoxes(best_idx,:);

I_letters = insertObjectAnnotation(poi_bin_image,'rectangle',bboxes,most_confident_letter);

imshow(I_letters);



