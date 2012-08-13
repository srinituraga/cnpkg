% load data
load ('/Users/viren/exhibit/0723/e1088_inout_validate4.mat');
input = reshape (im, [1 size(im)]);
offset = 8;
labels = labels (offset+1:end-offset, offset+1:end-offset, offset+1:end-offset);
output = reshape (labels, [1 size(labels)]);
label_mask = label_mask (offset+1:end-offset, offset+1:end-offset, offset+1:end-offset);
outputMask = reshape (label_mask, [1 size(label_mask)]);

p = struct;
%p.fCount  = [1 10 10 10 1];
p.fCount  = [1 1 1 1 1];
p.fSize   = [0 5 5 5 5];
p.fDepth  = [0 5 5 5 5];
p.eta     = [0 1 1 1 1] * 1e-3;
%p.outSize = [4 4 4];
p.outSize = [1 1 1];
p.iterations = 1;

p.fullOutSize = size (squeeze (output));
m = cnpkg_buildmodel(p);

%cns('test', m, 'gpu', 'mean');
cns('init', m, 'cpu', 'mean'); % run in emulation mode


input = rand(size (input)) * 0.01;
output = rand(size (output));
outputMask = rand(size (outputMask));

%sliceid = 100;
%input = shiftdim (im(:, :, sliceid), -1);
%output = labels(:, :, sliceid);
%validCrop = [sum(p.fSize) sum(p.fDepth)];
%correct = shiftdim (output (1+12:255-12, 1+12:255-12), -1);
indexes = zeros(3, p.iterations);
indexes(1, :) = randi(p.fullOutSize(1) - p.outSize(1) + 1, 1, p.iterations) - 1;
indexes(2, :) = randi(p.fullOutSize(2) - p.outSize(2) + 1, 1, p.iterations) - 1;
indexes(3, :) = randi(p.fullOutSize(3) - p.outSize(3) + 1, 1, p.iterations) - 1;

tic;
cns('set', 1, 'val', input);
cns('set', 2, 'val', output);
cns('set', 3, 'val', indexes);
cns('set', 4, 'val', outputMask);

%ip2 = cns('get', 1, 'val')

%cns('step', 1, 6);
%x1 = cns('get', 6, 'val');

cns('run', p.iterations);
    %ab = cns('get', 2, 'val', input);
    %pause;
%w7 = cns('get', 21, 'val');
toc;

%cns('done');