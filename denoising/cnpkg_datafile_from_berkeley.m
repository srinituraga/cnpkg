function cnpkg_datafile_from_berkeley
% converts noisy images from the berkeley dataset into a format that can be
% used for srini's cnpkg scripts

%% Parameters (change these!)

% cell array where each cell is the name of a file to load for the input
% set
input_files{1} = '/Users/viren/net_sets/berkeley/training_all/set1_im_1noise.mat';
input_files{2} = '/Users/viren/net_sets/berkeley/training_all/set1_im_15noise.mat';
input_files{3} = '/Users/viren/net_sets/berkeley/training_all/set1_im_25noise.mat';
input_files{4} = '/Users/viren/net_sets/berkeley/training_all/set1_im_50noise.mat';
input_files{5} = '/Users/viren/net_sets/berkeley/training_all/set1_im_100noise.mat';

% this can be one file or a list of same length as input_files
label_files{1} = '/Users/viren/net_sets/berkeley/training_all/set1_labels.mat';

% masks are automatically set to ones for the entire size of the label
% image if this is empty. right now you have no other choice.
mask_files = {};

% cell array where each cell is a list of images to include from the
% corresponding infile, above
imageLists = {1:40};

% name of file to save
savefile = '/Users/stetner/data/denoising/set1_im_allnoise_1to40.mat';

% if true, each image will be normalized to the range [0,1]
flags.normalize = 1;

%%

flags.reuse_labels = length(label_files)==1;
flags.auto_masks   = ~exist('mask_files','var') || isempty(mask_files);
flags.reuse_masks  = length(mask_files)==1;
flags.reuse_imageList = length(imageLists)==1;


%%

input = {};
label = {};
mask = {};
for n = 1:length(input_files)
    
    % pick the list of images to use from this file
    if flags.reuse_imageList
        imageList = imageLists{1};
    else
        imageList = imageLists{n};
    end
    
    % load input images
    raw = load(input_files{n});
    temp = matrix2block( raw.im, imageList, flags.normalize);
    input = horzcat(input, temp);
    
    % if we are doing separate label files, load labels too
    if ~flags.reuse_labels
        raw = load(label_files{n});
        temp = matrix2block( raw.labels, imageList, flags.normalize);
        label = horzcat(label, temp);
        
        maskMatrix = ones(size(raw.labels));
        maskMatrix = single(maskMatrix);
        temp = matrix2block( maskMatrix, imageList, flags.normalize);
        mask = horzcat(mask, temp);
        
        temp = (1:length(temp)) + length(label)-length(temp); %FIXME
        input2label = horzcat(input2label, temp);
    end
    
end

% if we are not doing separate label files, load labels from a single
% file at the end
if flags.reuse_labels
    raw = load(label_files{1});
    if flags.reuse_imageList
        imageList_label = imageLists{1};
    else
        imageList_label = union(imageLists{:});
    end
    label = matrix2block( raw.labels, imageList_label, flags.normalize );
    
    maskMatrix = ones(size(raw.labels));
    maskMatrix = single(maskMatrix);
    mask = matrix2block( maskMatrix, imageList_label, 0);
        
    if flags.reuse_imageList
        input2label = horzcat(repmat(imageLists{1},1,length(input_files)));
    else
        input2label = horzcat(imageLists{:});
    end
    
end

% for debugging:
showInputsAndLabels(input,label,input2label)

save(savefile, 'input','label','mask','input2label')
disp(['saved to ' savefile])

end

function block = matrix2block(M, slices,normalize)

block = cell(1,length(slices));

for n = 1:length(slices)
    z = slices(n);
    im = M(:,:,z);
    im = shiftdim(im,-1);
    if normalize
        im = (im - min(im(:)))/(max(im(:))-min(im(:)));
    end
    block{n} = im;
end

end

function showInputsAndLabels(input,label,input2label)
% shows each inputs with its corresponding label
% use this to check to make sure that inputs are mapped to labels correctly

for ni = 1:length(input)
    no = input2label(ni);
    
    subplot(1,2,1)
    imagesc(squeeze(input{ni}))
    colormap gray
    title('input')
    
    subplot(1,2,2)
    imagesc(squeeze(label{no}))
    colormap gray
    title('label')
    
    pause(0.2)
    
end
end