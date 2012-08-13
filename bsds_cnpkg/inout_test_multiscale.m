function [output]=inout_test_multiscale(m,scales,show_plot)

if ~exist('show_plot','var'),
	show_plot = 0;
end

%% -----------------------Initializing

offset = m.layers{m.layer_map.label}.offset;
m.params.minibatch_size = 1/2;
m.params.nIterPerEpoch = 1;

% load data
testing_set = load(m.data_info.testing_file,'input');
disp('Loaded data file.')
nInput = length(testing_set.input);
disp('Beginning training...');


%% ------------------------Testing
for k = 1:nInput,

	im = permute(testing_set.input{k},[2 3 1 4]);
	imSz = [size(im(:,:,1)) 1];
	output{k} = zeros(imSz([3 1 2]),'single');
	norm = zeros(imSz([3 1 2]),'single');

	for s = scales,

		imRescale = imresize(im,s);
		imSzRescale = [size(imRescale(:,:,1)) 1];
		inoutRescale = zeros(imSzRescale,'single');
		normRescale = zeros(imSzRescale,'single');

		% construct cns model for gpu
		m.params.output_size = imSzRescale - 2*offset;
		m = cnpkg_mapdim_layers(m);
		m.layers{m.layer_map.input}.val = permute(imRescale,[3 1 2 4]);

		% initialize gpu
		fprintf('Initializing on gpu...');tic
		cns('init',m,'gpu','mean')
		fprintf(' done. ');toc

		% run the gpu
		fprintf('running gpu...');tic
		inoutRescale(offset(1)+1:end-offset(1),offset(2)+1:end-offset(2)) =  ...
			squeeze(cns('step',m.step_map.fwd(1)+1,m.step_map.fwd(end),{m.layer_map.output,'val'}));
		fprintf(' done. ');toc

		output{k} = output{k} + permute(imresize(inoutRescale,imSz(1:2)),[3 1 2]);

		normRescale(offset(1)+1:end-offset(1),offset(2)+1:end-offset(2)) =  1;
		norm = norm + permute(imresize(normRescale,imSz(1:2)),[3 1 2]);
	end

	output{k} = output{k}./norm;
	output{k} = output{k}(:,offset(1)+1:end-offset(1),offset(2)+1:end-offset(2));

	if show_plot,
		figure(1)
		subplot(211), imagesc(permute(testing_set.input{k},[2 3 1 4])),axis image
		subplot(212), imagesc(permute(output{k},[2 3 1 4]),[0 1]),colormap gray,axis image
		drawnow
	end


end

return
