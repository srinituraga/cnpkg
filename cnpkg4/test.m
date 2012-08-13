function [output,leftBorder,rightBorder] = test(m,testing_set,zero_pad,show_plot)

if ischar(m),
	fprintf(['Loading network from ' m '...\n']);
	load(m,'m');
end

if ~exist('zero_pad','var') || isempty(zero_pad),
	zero_pad = false;
end

if ~exist('show_plot','var') || isempty(show_plot),
	show_plot = 0;
end

%% -----------------------Initializing

m.params.minibatch_size = 1/2;
m.params.nIterPerEpoch = 1;
m = cnpkg4.SetupForTesting(m);
m = cnpkg4.SetupStepNo(m,0);

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


%% ------------------------Testing
for k = 1:nInput,

	fprintf(['Processing image ' num2str(k) ' of size [' num2str(size(testing_set.input{k})) ']\n'])
	imSz = [size(testing_set.input{k},2) size(testing_set.input{k},3) size(testing_set.input{k},4)];
	output{k} = zeros([m.layers{m.layer_map.output}.size{1} imSz],'single');

	% construct cns model for gpu
	if zero_pad,
        m = cnpkg4.MapDimFromOutput(m,imSz,1);
	else,
        m = cnpkg4.MapDimFromOutput(m,imSz-m.totalBorder,1);
	end

	% store border sizes
	totalBorder{k} = cell2mat(m.layers{m.layer_map.input}.size(2:4)) - cell2mat(m.layers{m.layer_map.output}.size(2:4));
	leftBorder{k} = m.offset;
	rightBorder{k} = totalBorder{k} - leftBorder{k};

    m.layers{m.layer_map.input}.val = zeros(cell2mat(m.layers{m.layer_map.input}.size),'single');
	if zero_pad,
		m.layers{m.layer_map.input}.val(:, ...
					leftBorder{k}(1)+(1:imSz(1)), ...
					leftBorder{k}(2)+(1:imSz(2)), ...
					leftBorder{k}(3)+(1:imSz(3))) ...
						= testing_set.input{k};
		for j=1:3,
			outIdx{j} = 1:imSz(j);
		end
	else,
        imSz2 = min(imSz,cell2mat(m.layers{m.layer_map.input}.size(2:4)));
        rightBorder{k} = rightBorder{k} + imSz-cell2mat(m.layers{m.layer_map.input}.size(2:4));
		m.layers{m.layer_map.input}.val(:, ...
                    1:imSz2(1), ...
                    1:imSz2(2), ...
                    1:imSz2(3)) ...
                        = testing_set.input{k}(:, ...
                            1:imSz2(1), ...
                            1:imSz2(2), ...
                            1:imSz2(3));
		for j=1:3,
			outIdx{j} = (leftBorder{k}(j)+1):(imSz(j)-rightBorder{k}(j));
		end
	end

	% initialize gpu
	fprintf('Initializing on device...');tic
	cns('init',m)
	fprintf(' done. ');toc

	% run the gpu
	fprintf('running forward pass...');tic
	output{k}(:,outIdx{1},outIdx{2},outIdx{3}) = cns('run',1,{m.layer_map.output,'val'});
	fprintf(' done. ');toc

	if show_plot,
		try,
			figure(1)
			subplot(211), imagesc(permute(testing_set.input{k},[2 3 1 4])),axis image
			subplot(212), imagesc(permute(output{k},[2 3 1 4]),[0 1]),colormap gray,axis image
			drawnow
		catch,
		end
	end


end

cns done

return
