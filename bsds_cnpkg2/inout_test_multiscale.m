function [output]=inout_test_multiscale(m,testing_file,scales,zero_pad,show_plot)

if ~exist('zero_pad','var') || isempty(zero_pad),
	zero_pad = false;
end

if ~exist('show_plot','var') || isempty(show_plot),
	show_plot = false;
end

%% -----------------------Initializing

m.params.minibatch_size = 1/2;
m.params.nIterPerEpoch = 1;
m = cnpkg3_rebuildnet(m);

% load data
testing_set = load(testing_file,'input');
disp('Loaded data file.')
nInput = length(testing_set.input);
disp('Beginning testing...');


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
%		m.params.output_size = imSzRescale - 2*m.offset;
		if ~zero_pad,
			m = cnpkg3_mapdim_layers_fwd(m,imSzRescale,1);
			m.layers{m.layer_map.input}.val(:,:,:,:,1) = permute(imRescale,[3 1 2 4]);
% 			m.layers{m.layer_map.input}.val(:,:,:,:,2) = permute(flipdim(imRescale,2),[3 1 2 4]);
			inoutSz = cell2mat(m.layers{m.layer_map.output}.size(2:3));
			for kk=1:2,
				outputIdxRescale{kk} = m.offset(kk)+(1:inoutSz(kk));
			end
		else,
			m = cnpkg3_mapdim_layers_bkwd(m,imSzRescale,1);
			totalBorder = cell2mat(m.layers{m.layer_map.input}.size(2:4)) - cell2mat(m.layers{m.layer_map.output}.size(2:4));
			leftBorder = m.offset;
			rightBorder = totalBorder - leftBorder;
			m.layers{m.layer_map.input}.val = zeros(cell2mat(m.layers{m.layer_map.input}.size));
			m.layers{m.layer_map.input}.val(:, ...
					leftBorder(1)+(1:imSzRescale(1)), ...
					leftBorder(2)+(1:imSzRescale(2)), ...
					leftBorder(3)+(1:imSzRescale(3)),1) ...
										= permute(imRescale,[3 1 2 4]);
% 			m.layers{m.layer_map.input}.val(:, ...
% 					leftBorder(1)+(1:imSzRescale(1)), ...
% 					leftBorder(2)+(1:imSzRescale(2)), ...
% 					leftBorder(3)+(1:imSzRescale(3)),2) ...
% 										= permute(flipdim(imRescale,2),[3 1 2 4]);
			for kk=1:2,
				outputIdxRescale{kk} = (1:imSzRescale(kk));
			end
		end

		% initialize gpu
		fprintf('Initializing on gpu...');tic
		cns('init',m,'gpu','mean')
		fprintf(' done. ');toc

		% run the gpu
		fprintf('running gpu...');tic
		inoutRescale(outputIdxRescale{1},outputIdxRescale{2}) =  ...
			squeeze(cns('step',m.step_map.fwd(1)+1,m.step_map.fwd(end),{m.layer_map.output,'val'}));
% 		inoutRescale = (inoutRescale(:,:,1)+flipdim(inoutRescale(:,:,2),2))/2;
		fprintf(' done. ');toc
		normRescale(outputIdxRescale{1},outputIdxRescale{2}) =  1;

		output{k} = output{k} + permute(imresize(inoutRescale,imSz(1:2)),[3 1 2]);
		norm = norm + permute(imresize(normRescale,imSz(1:2)),[3 1 2]);
	end

	output{k} = output{k}./norm;
%	output{k} = output{k}(:,m.offset(1)+1:end-m.offset(1),m.offset(2)+1:end-m.offset(2));

	if show_plot,
		try,
			figure(1)
			subplot(211), imagesc(permute(testing_set.input{k},[2 3 1 4])),axis image
			title(['net: ' num2str(m.ID) ', im: ' num2str(k)])
			subplot(212), imagesc(1-permute(output{k},[2 3 1 4]),[0 1]),colormap gray,axis image
			drawnow
		catch,
		end
	end


end

return
