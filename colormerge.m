clear; clc; clf; close all;

im = imread('1.png');
% im = imread('targets_01.png');
im = imresize(im,[1080,1920]);
% im = im(500:560,1290:1330,:); %3.png
% im = im(650:770,1200:1335,:);%2.png
im = im(300:390,730:875,:);%1.png
% im = im(1510:1550,1465:1515,:);



% iterative bilateral filter
patch = im;
patchVar = std2(patch)^2;
DoS = patchVar/20;
spat_sigma = 8;
for j = 1:20
    im = imbilatfilt(im,DoS, spat_sigma);
end

% im = round(im ./ 10 ) * 10;

% im = rgb2lab(im);

R = reshape(im(:,:,1),[],1);
G = reshape(im(:,:,2),[],1);
B = reshape(im(:,:,3),[],1);

col = [R,G,B];

[colu,m,n] = unique(col,'rows');
color_counts = accumarray(n, 1);


[color_counts, id] = sort(color_counts);
uc=colu(id,:);
uc = flipud(uc);
color_counts = flipud(color_counts);

color1 = uc(1,:);
color2 = [-1,-1,-1];
color3 = [-1,-1,-1];


thresh = 30;

i = 2;
while(eudist(color1,uc(i,:)) < thresh)
   i = i + 1; 
   
end
disp(i);

if(i < size(uc,1))
    color2 = uc(i,:);
    
    disp(i)
    i = i + 1;
    if(i < size(uc,1))
        while((eudist(color1,uc(i,:)) < thresh) || (eudist(color2,uc(i,:)) < thresh))
            i = i + 1;
        end
        disp(i)
        if(i < size(uc,1))
            color3 = uc(i,:);
        end
    end
    
else
    
end
% color1 = lab2rgb(color1)*255;
% color2 = lab2rgb(color2)*255;
% color3 = lab2rgb(color3)*255;
% 
% x = [0 1 1 0] ; y = [0 0 1 1] ;
% figure
% subplot(1,3,1);
% fill(x,y,color1/255)
% subplot(1,3,2);
% fill(x,y,double(color2)/255)
% subplot(1,3,3);
% fill(x,y,double(color3)/255)

function out = eudist(v1,v2)
    out = norm(double(v1)-double(v2));
end









