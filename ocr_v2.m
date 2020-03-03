clear; clc;

imnum = 1;
imname = ["PARTIEEE_output_images/f01_bf/img",num2str(imnum),".png"];
imname = strjoin(imname,"");
im = imread(imname);

colname = ["PARTIEEE_output_images/f01_colors/",num2str(imnum),".txt"];
colname = strjoin(colname,"");
col3 = csvread(colname);

c1 = double(col3(1,:));
c2 = double(col3(2,:));
c3 = double(col3(3,:));

im = imrotate(im,170);

masks = 0.* im;

[m,n,d] = size(im);
for i=1:m
    for j=1:n
        p = double([im(i,j,1),im(i,j,2),im(i,j,3)]);
        d1 = distance(c1,p);
        d2 = distance(c2,p);
        d3 = distance(c3,p);
        d = [d1,d2,d3];
        
        [m,mindex] = min(d);
        masks(i,j,mindex) = 1;
        
    end
end

l1_count = sum(sum(masks(:,:,1)));
l2_count = sum(sum(masks(:,:,2)));
l3_count = sum(sum(masks(:,:,3)));
lcount = [l1_count,l2_count,l3_count];
[~,layer_max_idx] = max(lcount);

masks = 255*masks;

% masks = imgaussfilt(masks,1);



bbox_im = im;

letters_detected = [];
confidences = [];

for i=1:3
       
    bw = masks(:,:,i);

    cc = bwconncomp(bw);
    
    while(true)
        numPix = cellfun(@numel,cc.PixelIdxList);
        [smallest,idx] = min(numPix);
        if(smallest < 0.25*sum(numPix))
            bw(cc.PixelIdxList{idx}) = 0;
            cc = bwconncomp(bw);             
        else
            break
        end

    end
    
%     while(cc.NumObjects > 1)
%         numPix = cellfun(@numel,cc.PixelIdxList);
%         [smallest,idx] = min(numPix);
%         bw(cc.PixelIdxList{idx}) = 0;
%         cc = bwconncomp(bw);
%     end

    masks(:,:,i) = bw;
%     corrected = imtophat(bw,strel('disk',15));
%     marker = imerode(corrected,strel('line',10,0));
%     clean = imreconstruct(marker,corrected);
%     
%     bin = imbinarize(clean);
%     bin_image = 255*bin;
    
    O = ocr(bw,'CharacterSet','ABCDEFGHIJKLMNOPQRSTUVWXYZ','TextLayout','Block');
    [sortedConf,sortedIndex] = sort(O.CharacterConfidences,'descend');
    idx_nan_removed = sortedIndex(~isnan(sortedConf));
    if(size(idx_nan_removed) ~= 0)
        best_idx = idx_nan_removed(1);
        most_confident_letter = num2cell(O.Text(best_idx));
        

        
        most_confident_letter = num2cell(O.Text(best_idx));
        bboxes = O.CharacterBoundingBoxes(best_idx,:);

        bbox_im = insertObjectAnnotation(bbox_im,'rectangle',bboxes,most_confident_letter);

        if(max(sortedConf) > 0.4 && i~= layer_max_idx)
           letters_detected = [letters_detected,most_confident_letter];
           confidences = [confidences,max(sortedConf)];
        end
    end


end

imshow(masks);

letters_detected
confidences

function out = distance(v1,v2)
    out = norm(v1-v2);
end