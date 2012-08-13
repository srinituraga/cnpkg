fprintf('Defining a network model....\n');

m = struct;

m.package = 'demopkg';

m.layers{1}.type    = 'input';
m.layers{1}.pz      = 0;
m.layers{1}.size{1} = 1;
m = cns_mapdim(m, 1, 'y', 'pixels', 256);
m = cns_mapdim(m, 1, 'x', 'pixels', 256);

m.layers{2}.type    = 'scale';
m.layers{2}.pz      = 1;
m.layers{2}.size{1} = 1;
m = cns_mapdim(m, 2, 'y', 'scaledpixels', 256, 2);
m = cns_mapdim(m, 2, 'x', 'scaledpixels', 256, 2);

m.layers{3}.type    = 'filter';
m.layers{3}.pz      = 2;
m.layers{3}.rfCount = 11;
m.layers{3}.fParams = {'gabor', 0.3, 5.6410, 4.5128};
m.layers{3}.size{1} = 4;
m = cns_mapdim(m, 3, 'y', 'int', 2, 11, 1);
m = cns_mapdim(m, 3, 'x', 'int', 2, 11, 1);

%***********************************************************************************************************************

% uncomment the following line if you don't have a GPU:
% cns('platform', 'cpu');

fprintf('Building the model (platform: %s)....\n', upper(cns('platform')));

cns('init', m);

%***********************************************************************************************************************

fprintf('Reading a test image and loading it into the model....\n');

in = imread(fullfile(fileparts(mfilename('fullpath')), 'ketch_0010.jpg'));
in = single(in) / single(intmax(class(in)));
in = shiftdim(in, -1);

cns('set', 1, 'val', in);

%***********************************************************************************************************************

fprintf('Running the model and retrieving result....\n');

tic;

cns('run');
res = cns('get', 3, 'val');

toc;

%***********************************************************************************************************************

fprintf('Plotting result....\n');

figure;
subplot(2, 3, 1); imshow(shiftdim(in, 1));
subplot(2, 3, 2); imagesc(shiftdim(res(1, :, :), 1));
subplot(2, 3, 3); imagesc(shiftdim(res(2, :, :), 1));
subplot(2, 3, 5); imagesc(shiftdim(res(3, :, :), 1));
subplot(2, 3, 6); imagesc(shiftdim(res(4, :, :), 1));

%***********************************************************************************************************************

fprintf('Releasing model resources....\n');

cns('done');