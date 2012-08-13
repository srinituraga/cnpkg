function [m]=learn(m,show_plot)


%% -----------------------Initializing
% configure IO
% configure save directory
if(isfield(m.params,'save_directory') && exist(m.params.save_directory,'dir'))
	m.params.save_string=[m.params.save_directory, num2str(m.ID),'/'];
	disp(['saving network to directory: ', m.params.save_string]); 
	mkdir(m.params.save_string);
else
	error('save directory not found!');
end

if ~exist('show_plot','var'),
	show_plot = 0;
end

% some minimax params
if ~isfield(m.params,'constrained_minimax'),
	m.params.constrained_minimax = true;
end

% initialize statistics
% initialize clocks
if(~isfield(m.stats, 'last_backup'))
	m.stats.last_backup=clock;
end
if(~isfield(m.stats,'times'))
	m.stats.times=zeros(5000,1);
end
if ~isfield(m.stats,'loss'),
	m.stats.loss = zeros(m.params.maxIter,1,'single');
	m.stats.classerr = zeros(m.params.maxIter,1,'single');
else,
	m.stats.loss(m.params.maxIter) = 0;
	m.stats.classerr(m.params.maxIter) = 0;
end

% initialize random number generators
rand('state',sum(100*clock));
randn('state',sum(100*clock));

% init counters
if ~isfield(m.stats,'iter'),
	m.stats.iter = 0;
end
if ~isfield(m.stats,'epoch'),
	m.stats.epoch = 0;
end
cnpkg_log_message(m, ['initialized stats']);

% construct cns model for gpu
m = cnpkg_mapdim_layers(m);
offset = m.layers{m.layer_map.label}.offset;
totalBorder = 2*offset + cell2mat(m.layers{m.layer_map.output}.size(2:4)) - 1;
halfTotalBorder = ceil(totalBorder/2);

% load data
training_set = load(m.data_info.training_file);
cnpkg_log_message(m,['Loaded data file...']);

nInput = length(training_set.input);
nLabel = length(training_set.label);
m.layers{m.layer_map.input}.inputblock = training_set.input;
m.layers{m.layer_map.label}.labelblock = training_set.label;
m.layers{m.layer_map.label}.maskblock = training_set.mask;

m.layers{m.layer_map.minibatch_index}.size = {5,m.params.nIterPerEpoch,m.params.minibatch_size*2};
index = zeros(cell2mat(m.layers{m.layer_map.minibatch_index}.size));
m.layers{m.layer_map.minibatch_index}.val = index;

% initialize gpu
cnpkg_log_message(m,'Initializing on gpu...');tic
cns('init',m,'gpu','mean')
cnpkg_log_message(m,' done. ');toc


% done initializing!
cnpkg_log_message(m, ['initialization complete!']);



%% ------------------------Training
cnpkg_log_message(m, ['beginning training.. ']);
while(m.stats.iter < m.params.maxIter),

	%% Assemble minibatch indices ---------------------------------------------------------------
fprintf('Assembling minibatch indices...');tic

% build index. This tells the network what to do on each iteration. Srini
% says:
% index is now 5 x nIter x nPatchPerMinibatch
% for each iter & patch, the 5 x 1 x 1 index is as follows:
% (1 2 3) are spatial coordinates to the corner of the patch (not the
% center!)
% 4 is the index of the input image for a patch
% 5 is the index of the label & mask images for this patch
% remember! index is zero-based!

% find the size of each input and output patch
for k_in = 1:length(training_set.input)
    % for each image in the input block
    k_out = training_set.input2label(k_in);
    
        [trash,xsize,ysize,dsize] = size(training_set.label{k_out});
        
        xmax(k_in) = xsize - m.layers{m.layer_map.input}.size{2} + 1;
        ymax(k_in) = ysize - m.layers{m.layer_map.input}.size{3} + 1;
        dmax(k_in) = dsize - m.layers{m.layer_map.input}.size{4} + 1;
        
end

% randomly generate index
iIn  = randsample(length(training_set.input), m.params.nIterPerEpoch*m.params.minibatch_size*2, true);
iOut = training_set.input2label(iIn)';
iX   = ceil(rand(size(iIn)) .* xmax(iIn)');
iY   = ceil(rand(size(iIn)) .* ymax(iIn)');
iD   = ceil(rand(size(iIn)) .* dmax(iIn)');

index = reshape([iX iY iD iIn iOut]',5,m.params.nIterPerEpoch,m.params.minibatch_size*2);

fprintf('done. '); toc

	%% Train ------------------------------------------------------------------------------------

    fprintf('Training on gpu...');tic
	cns('set',{0,'iter_no',0},{m.layer_map.minibatch_index,'val',index-1});
    loss = cns('run',m.params.nIterPerEpoch,{m.layer_map.output,'loss'});
fprintf('done. ');toc

	if m.params.debug >= 2,
		cnpkg_log_message(m,['DEBUG_MODE: loss: ' num2str(sum(loss(:)))]);
	end
	m.stats.loss(m.stats.iter+(1:m.params.nIterPerEpoch)) = mean(loss(:,:),2);
	m.stats.classerr(m.stats.iter+(1:m.params.nIterPerEpoch)) = 0; % FIXME

	%% Update counters --------------------------------------------------------------------------
	m.stats.iter = m.stats.iter+m.params.nIterPerEpoch;
	m.stats.epoch = m.stats.epoch+1;

	%% Save current state ----------------------------------------------------------------------
	if (etime(clock,m.stats.last_backup)>m.params.backup_interval) || (m.stats.iter==length(m.stats.times)),
		cnpkg_log_message(m, ['Saving network state... Iter: ' num2str(m.stats.iter)]);
		m = cns('update',m);
		m.layers{m.layer_map.input}.inputblock = [];
		m.layers{m.layer_map.label}.labelblock = [];
		m.layers{m.layer_map.label}.maskblock = [];
		save([m.params.save_string,'latest'],'m');
		m.stats.last_backup = clock;
	end

	if ~rem(m.stats.epoch,m.params.nEpochPerSave),
		%% save current state/statistics
		m = cns('update',m);
		m.layers{m.layer_map.input}.inputblock = [];
		m.layers{m.layer_map.label}.labelblock = [];
		m.layers{m.layer_map.label}.maskblock = [];
		save([m.params.save_string,'epoch-' num2str(m.stats.epoch)],'m');
		cnpkg_log_message(m,['Epoch: ' num2str(m.stats.epoch) ', Iter: ' num2str(m.stats.iter) '; Loss ' num2str(mean(loss(:)))]);
		try,
			figure(1)
			plot(smooth(m.stats.loss(1:m.stats.iter),m.params.nIterPerEpoch))
			drawnow
		catch,
			warning('unable to plot')
		end
	end

	if show_plot,
		try,
			ii = cns('get',{m.layer_map.input,'val'});
			ii = permute(ii,[2 3 1 5 4]);
			ll = cns('get',{m.layer_map.label,'val'});
			ll = permute(ll,[2 3 1 5 4]);
			oo = cns('get',{m.layer_map.output,'val'});
			oo = permute(oo,[2 3 1 5 4]);
	[ll(:)';oo(:)'];
			figure(2)
			smpl = randsample(m.params.minibatch_size*2,1,1);
			subplot(311), imagesc(ii(offset(1)+1:end-offset(1), ...
										offset(2)+1:end-offset(2), ...
										:,smpl))
			subplot(312), imagesc(ll(:,:,smpl),[0 1]),colormap gray
			subplot(313), imagesc(oo(:,:,smpl),[0 1]),colormap gray
			drawnow
		catch,
			warning('unable to plot')
		end
	end

end

return


return
