% Run this code after extracting the "extra" tar images and digitStruct.mat
% file to a folder named "extra" inside "data" folder (./data/extra -->
% here put this file.)

% This file is the first file to be run in my implementation.
% After running this file, you'll get ".mat" files in directory
% ./data/extra/mat_files:
% Each file contains the bounding boxes corresponding to each digit in
% current image. The filename of each ".mat" file corresponds to the
% filename of the image from which I extracted the bounding boxes.

%The bounding boxes are extracted using image processing techniques useful
%for text detection, such as MSER feature detection and Stroke Width
%variance.

CROP_SIZE = 16;     % The size of the crop for each digit
MAX_DIGITS_PER_IMG = 5;
fileList = dir('*.png');
[num_files, ~] = size(fileList);
all_digits_arrays = zeros(num_files, MAX_DIGITS_PER_IMG, CROP_SIZE, CROP_SIZE);
all_digits_labels = zeros(num_files, MAX_DIGITS_PER_IMG);
currentFolder = pwd;

for file = fileList'
    cur_img = imread(file.name);
    cur_filename = file.name;
    I = rgb2gray(cur_img);
    
    % Run the digits detection algorithm on all images in the directory.
    % Skip image if there's not enough contrast and the algorithm fails
    % because of this.
    
    % Un-comment the sections that displays the image in each step if
    % you want to check each step's functionallity for digit detection.
    
    try

        % Detect MSER regions.
        [mserRegions, mserConnComp] = detectMSERFeatures(I, ... 
            'RegionAreaRange',[200 8000],'ThresholdDelta',4);
        
        %{
        figure
        imshow(I)
        hold on
        plot(mserRegions, 'showPixelList', true,'showEllipses',false)
        title('MSER regions')
        hold off
        %}

        % Use regionprops to measure MSER properties
        mserStats = regionprops(mserConnComp, 'BoundingBox', 'Eccentricity', ...
            'Solidity', 'Extent', 'Euler', 'Image');

        % Compute the aspect ratio using bounding box data.
        bbox = vertcat(mserStats.BoundingBox);
        %disp(bbox)
        w = bbox(:,3);
        h = bbox(:,4);
        aspectRatio = w./h;

        % Threshold the data to determine which regions to remove. These thresholds
        % may need to be tuned for other images.
        filterIdx = aspectRatio' > 3; 
        filterIdx = filterIdx | [mserStats.Eccentricity] > .995 ;
        filterIdx = filterIdx | [mserStats.Solidity] < .3;
        filterIdx = filterIdx | [mserStats.Extent] < 0.2 | [mserStats.Extent] > 0.9;
        filterIdx = filterIdx | [mserStats.EulerNumber] < -4;

        % Remove regions
        mserStats(filterIdx) = [];
        mserRegions(filterIdx) = [];

        % Show remaining regions
        %{
        figure
        imshow(I)
        hold on
        plot(mserRegions, 'showPixelList', true,'showEllipses',false)
        title('After Removing Non-Text Regions Based On Geometric Properties')
        hold off
        %}
        
        % Get a binary image of the a region, and pad it to avoid boundary effects
        % during the stroke width computation.
        regionImage = mserStats(6).Image;
        regionImage = padarray(regionImage, [1 1]);

        % Compute the stroke width image.
        distanceImage = bwdist(~regionImage); 
        skeletonImage = bwmorph(regionImage, 'thin', inf);

        strokeWidthImage = distanceImage;
        strokeWidthImage(~skeletonImage) = 0;

        % Show the region image alongside the stroke width image.
        %{
        figure
        subplot(1,2,1)
        imagesc(regionImage)
        title('Region Image')

        subplot(1,2,2)
        imagesc(strokeWidthImage)
        title('Stroke Width Image')
        %}
        
        % Compute the stroke width variation metric 
        strokeWidthValues = distanceImage(skeletonImage);   
        strokeWidthMetric = std(strokeWidthValues)/mean(strokeWidthValues);

        % Threshold the stroke width variation metric
        strokeWidthThreshold = 0.4;
        strokeWidthFilterIdx = strokeWidthMetric > strokeWidthThreshold;

        % Process the remaining regions
        for j = 1:numel(mserStats)

            regionImage = mserStats(j).Image;
            regionImage = padarray(regionImage, [1 1], 0);

            distanceImage = bwdist(~regionImage);
            skeletonImage = bwmorph(regionImage, 'thin', inf);

            strokeWidthValues = distanceImage(skeletonImage);

            strokeWidthMetric = std(strokeWidthValues)/mean(strokeWidthValues);

            strokeWidthFilterIdx(j) = strokeWidthMetric > strokeWidthThreshold;

        end

        % Remove regions based on the stroke width variation
        mserRegions(strokeWidthFilterIdx) = [];
        mserStats(strokeWidthFilterIdx) = [];

        % Show remaining regions
        %{
        figure
        imshow(I)
        hold on
        plot(mserRegions, 'showPixelList', true,'showEllipses',false)
        title('After Removing Non-Text Regions Based On Stroke Width Variation')
        hold off
        %}

        %Bounding boxes separately:
        % Get bounding boxes for all the regions
        bboxes = vertcat(mserStats.BoundingBox);

        % Convert from the [x y width height] bounding box format to the [xmin ymin
        % xmax ymax] format for convenience.
        xmin = bboxes(:,1);
        ymin = bboxes(:,2);
        xmax = xmin + bboxes(:,3) - 1;
        ymax = ymin + bboxes(:,4) - 1;

        % Expand the bounding boxes by a small amount.
        expansionAmount = 0.02;
        xmin = (1-expansionAmount) * xmin;
        ymin = (1-expansionAmount) * ymin;
        xmax = (1+expansionAmount) * xmax;
        ymax = (1+expansionAmount) * ymax;

        % Clip the bounding boxes to be within the image bounds
        xmin = max(xmin, 1);
        ymin = max(ymin, 1);
        xmax = min(xmax, size(I,2));
        ymax = min(ymax, size(I,1));



        % Show the expanded bounding boxes
        expandedBBoxes = [xmin ymin xmax-xmin+1 ymax-ymin+1];
        %disp(size(expandedBBoxes));
        expandedBBoxes = unique(expandedBBoxes,'rows');
        %disp(size(expandedBBoxes));
        %disp((expandedBBoxes));
        non_overlap_bboxes = expandedBBoxes(1, :);
        
        % Filter out the duplicates of each bounding box / bounding boxes
        % that are just a slight deviations of the wanted bounding box
        for i = 1 : length(expandedBBoxes) - 1
            cur_row_diff = expandedBBoxes(i+1, :) - expandedBBoxes(i, :);
            large_jump = cur_row_diff > 5;
            if sum(large_jump) >= 1
                non_overlap_bboxes(end + 1, :) = expandedBBoxes(i+1, :);
            end

        end
        %disp(non_overlap_bboxes);
        
    catch
        % the filenames displayed don't match our algorithm and the
        % assumption that there's a high contrast between the text and
        % the background.
        disp(cur_filename);
        continue;
    end
        
        %Saving for each filename the locations of the bounding boxes (the variable
        % contains the locations of all bounding boxes in the image)
        current_path = pwd;

        output_path = strcat(current_path, '\mat_files\');
        [filepath, name, ext] = fileparts(cur_filename); 

        output_path = strcat(output_path, name);
        output_path = strcat(output_path, '.mat');
        % save variable for each filename in the output mat file
        save(output_path, 'non_overlap_bboxes' ); 
      
end