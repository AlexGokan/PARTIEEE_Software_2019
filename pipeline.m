clear; clc;

listing = dir('PARTIEEE_output_images');
num_files = size(listing,1);


latest_frame_name = listing(num_files).name;
detected_filenames = dir(['PARTIEEE_output_images/',latest_frame_name]);
detected_filenames = detected_filenames(3:end);
num_files = size(detected_filenames,1);

for mser_num = 6
   color_cmd = ['python gmm_test.py ',latest_frame_name,' ',num2str(mser_num)]; 
   [color_status,color_cmd_out] = system(color_cmd);
   colors = splitlines(color_cmd_out);
    c1 = colors(1);
    c2 = colors(3);
    c3 = colors(5);
   
    c1str = sprintf('%s,',c1{:});
    c1num = sscanf(c1str,'%g,',[3,inf]).';
   
    c2str = sprintf('%s,',c2{:});
    c2num = sscanf(c2str,'%g,',[3,inf]).';
    
    c3str = sprintf('%s,',c3{:});
    c3num = sscanf(c3str,'%g,',[3,inf]).';
    
    c1 = double(c1num)
    c2 = double(c2num)
    c3 = double(c3num)
    
    imname = ["PARTIEEE_output_images/",latest_frame_name,"/img",num2str(mser_num),".png"];
    imname = strjoin(imname,"");
    imname = convertStringsToChars(imname);
    im = imread(imname);
    
%     im = imsharpen(im);
%     im = imsharpen(im);
    
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

    
   
end

function out = distance(v1,v2)
    out = norm(v1-v2);
end

