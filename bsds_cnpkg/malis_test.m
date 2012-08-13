function [output]=learn(m,show_plot,nScale)

if ~exist('show_plot','var') || isempty(show_plot),
	show_plot = 0;
end

if ~exist('nScale','var') || isempty(nScale),
	nScale = 1;
end

%% -----------------------Initializing

% % set up the layer step nos
% for l = 1:length(m.layers),
% % 	m.layers{l}=rmfield(m.layers{l},'stepNo');
% 	m.layers{l}.stepNo = [];
% end
% step = 0;
% for l = m.layer_map.hidden,
% 	step = step+1;
% 	m.layers{l}.stepNo = step;
% end
% step = step+1;
% m.layers{m.layer_map.output}.stepNo = step;
% step = step+1;
% for l = 1:length(m.layers),
% 	if isempty(m.layers{l}.stepNo),
% 		m.layers{l}.stepNo = step;
% 	end
% end

if ~isfield(m.layer_map,'label'),
	offset = m.layers{m.layer_map.output}.offset;
else,
	offset = m.layers{m.layer_map.label}.offset;
end
m.params.minibatch_size = 1/2;
m.params.nIterPerEpoch = 1;

% load data
testing_set = load(m.data_info.testing_file,'input');
disp('Loaded data file.')
nInput = length(testing_set.input);
disp('Beginning training...');


%% ------------------------Testing
for k = 1:nInput,

	imSz = [size(testing_set.input{k},2) size(testing_set.input{k},3) size(testing_set.input{k},4)];

	% construct cns model for gpu
	m.params.output_size = imSz - 2*offset;
	m = malis_mapdim_layers(m);
	m.layers{m.layer_map.input}.val = testing_set.input{k};
	size(testing_set.input{k})

	% initialize gpu
	fprintf('Initializing on gpu...');tic
	cns('init',m,'gpu','mean')
	fprintf(' done. ');toc

	% run the gpu
	fprintf('running gpu...');tic
	output{k} = cns('step',m.step_map.fwd(1)+1,m.step_map.fwd(end),{m.layer_map.output,'val'});
	fprintf(' done. ');toc

	if show_plot,
		figure(1)
		subplot(211), imagesc(permute(testing_set.input{k},[2 3 1 4])),axis image
		subplot(212), imagesc(permute(output{k},[2 3 1 4]),[0 1]),colormap gray,axis image
		drawnow
	end


end

return


return
