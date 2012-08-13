function [output]=test_split(m, testing_set, split_size)

if ischar(m),
	fprintf(['Loading network from ' m '...\n']);
	load(m,'m');
	if ~isequal(m.package,'cnpkg4'),
		warning(['Model trained using ' m.package '. Attempting to test with cnpkg4.'])
		m.package = 'cnpkg4';
	end
end

if ~exist('split_size','var') || isempty(split_size),
	split_size = [1 1 1]*100;
end

m.params.minibatch_size = 1/2;
m.params.nIterPerEpoch = 1;
m = cnpkg4.SetupForTesting(m);
[m, lastStep] = cnpkg4.SetupStepNo(m,0);

% load data
if ~exist('testing_set','var') || isempty(testing_set),
	fprintf(['Loading test data from ' m.data_info.testing_file '... ']);
	testing_set = load(m.data_info.testing_file,'input');
	fprintf([num2str(length(testing_set.input)) ' images found.\n']);
elseif ischar(testing_set),
	fprintf(['Loading test data from ' testing_set '... ']);
	testing_set = load(testing_set,'input');
	fprintf([num2str(length(testing_set.input)) ' images found.\n']);
elseif isnumeric(testing_set),
	im = testing_set; clear testing_set
	testing_set.input{1} = im; clear im
	fprintf(['Testing 1 image passed as input argument.\n']);
elseif iscell(testing_set),
	im = testing_set; clear testing_set
	testing_set.input = im; clear im
	fprintf(['Testing ' num2str(length(testing_set.input)) ' image(s) passed as input argument.\n']);
else
	error('unknown input. 2nd argument should be the test set. can be either 1) a file with a cell array called input, 2) a cell array of images or 3) a single image')
end
disp('Loaded data file.')
nInput = length(testing_set.input);
disp('Beginning testing...');

% construct cns model for gpu
m = cnpkg4.MapDimFromOutput(m,split_size,1);
% initialize gpu
fprintf(['Initializing device...']),tic
cns('init',m)
fprintf(' done. ');toc

% go through each image in the input set
for k = 1:nInput,

	imSz = [size(testing_set.input{k},2) size(testing_set.input{k},3) size(testing_set.input{k},4)];	
	bb = [1 size(testing_set.input{k},2); 1 size(testing_set.input{k},3); 1 size(testing_set.input{k},4)];
	output{k} = zeros([m.layers{m.layer_map.output}.size{1} imSz],'single');
	
	% generate split points
	block_bb = generate_splitpoints(bb, split_size+m.totalBorder, m.offset);
	if m.layers{m.layer_map.input}.size{1}>1,
		block_bb(4,1,:) = 1;
		block_bb(4,2,:) = m.layers{m.layer_map.input}.size{1};
	end
	nBlock = size(block_bb,3);

	% go through each split box for this image
	for j=1:size(block_bb,3),

		% run the gpu
		fprintf(['Processing block ' num2str(j) '/' num2str(nBlock) '...']);tic	
		cns('set',{m.layer_map.input,'val',testing_set.input{k}(:,block_bb(1,1,j):block_bb(1,2,j), block_bb(2,1,j):block_bb(2,2,j), block_bb(3,1,j):block_bb(3,2,j))});
		output{k}(:, ...
			double(block_bb(1,1,j))-1+m.offset(1)+(1:m.layers{m.layer_map.output}.size{2}), ...
			double(block_bb(2,1,j))-1+m.offset(2)+(1:m.layers{m.layer_map.output}.size{3}), ...
			double(block_bb(3,1,j))-1+m.offset(3)+(1:m.layers{m.layer_map.output}.size{4})) ...
				= cns('step',1,lastStep,{m.layer_map.output,'val'});
		fprintf(' done. ');toc
	end

end

cns done

return
