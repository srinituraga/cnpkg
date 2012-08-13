function [m]=learn(m)

% load the network if passed a filename
if(ischar(m))
	load(m,'m');
end


%% -----------------------Initializing
% configure IO
% configure save directory
if(isfield(m.params,'save_directory') && exist(m.params.save_directory,'dir'))
	m.params.save_string=[m.params.save_directory, num2str(m.ID),'/'];
	log_message([],['saving network to directory: ', m.params.save_string]); 
	mkdir(m.params.save_string);
else
	log_message([],'warning! save directory not found, using /tmp/');
	m.params.save_string=['/tmp/'];
	mkdir(m.params.save_string);
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

maxEpoch = ceil(m.params.maxIter/m.params.nIterPerEpoch);
if ~isfield(m.stats,'epoch_loss'),
	m.stats.epoch_loss = zeros(m.params.nIterPerEpoch,1,'single');
	m.stats.epoch_classerr = zeros(m.params.nIterPerEpoch,1,'single');
else,
	m.stats.epoch_loss(m.params.nIterPerEpoch) = 0;
	m.stats.epoch_classerr(m.params.nIterPerEpoch) = 0;
end
if ~isfield(m.stats,'loss'),
	m.stats.loss = zeros(maxEpoch,1,'single');
	m.stats.classerr = zeros(maxEpoch,1,'single');
else,
	m.stats.loss(maxEpoch) = 0;
	m.stats.classerr(maxEpoch) = 0;
end

% initialize random number generators
rand('state',sum(100*clock));
randn('state',sum(100*clock));

% init counters
if ~isfield(m.stats,'iter') || m.stats.iter < 1,
	m.stats.iter = 1;
end
if ~isfield(m.stats,'epoch') || m.stats.epoch < 1,
	m.stats.epoch = 1;
end
log_message(m, ['initializing stats']);

% construct cns model for gpu
m = malis_mapdim_layers(m,m.params.graph_size,m.params.minibatch_size*2);

% load data
load(m.data_info.training_files{1});
log_message(m,['Loaded data file...']);
% idx = randsample(length(im),2,1);
% im = im(idx); seg = seg(idx); mask = mask(idx);

% reformat 'im', apply mask, compute conn
for k = 1:length(im),
	imSz{k} = size2(im{k},1:3);
	imSzValid{k} = size2(im{k},1:3)-m.totalBorder-m.params.graph_size+1;
	im{k} = permute(im{k},[4 1 2 3]);
end
m.inputblock = im;
big_output_size = cell2mat(m.layers{m.layer_map.bigpos_output}.size(2:4));
half_big_output_size = floor(big_output_size/2);
total_offset = cell2mat(m.layers{m.layer_map.bigpos_input}.size(2:4))-1;
fprintf('Initializing on device...'),tic
cns('init',m)
fprintf(' done. '),toc


% done initializing!
log_message(m, ['initialization complete!']);



%% ------------------------Training
log_message(m, ['beginning training.. ']);
index_bigpos = zeros(cell2mat(m.layers{m.layer_map.bigpos_minibatch_index}.size));
index_bigneg = zeros(cell2mat(m.layers{m.layer_map.bigneg_minibatch_index}.size));
index = zeros(cell2mat(m.layers{m.layer_map.minibatch_index}.size));
mblabel = zeros(cell2mat(m.layers{m.layer_map.label}.size));
mbmask = zeros(cell2mat(m.layers{m.layer_map.label}.size));
while(m.stats.iter < m.params.maxIter),

	%% Assemble a minibatch ---------------------------------------------------------------------
	%% select a nice juicy image and image patch

	% Pick big patch for negative examples
	imPickNeg = randsample(length(im),1);
	segPickNeg = randsample(length(seg{imPickNeg}),1);

	for k=1:3,
		index_bigneg(k) = randi(imSz{imPickNeg}(k)-m.totalBorder(k)-big_output_size(k)+1,1);
	end
	index_bigneg(4) = imPickNeg;
	for k=1:3,
		idxGraphOutNeg{k} = (index_bigneg(k)+m.leftBorder(k)-1)+(1:big_output_size(k));
	end
	segNeg = single(connectedComponents( ...
				MakeConnLabel( ...
					seg{imPickNeg}{segPickNeg}(idxGraphOutNeg{1},idxGraphOutNeg{2},idxGraphOutNeg{3}), ...
					m.params.nhood),m.params.nhood));
	maskNeg = mask{imPickNeg}{segPickNeg}(idxGraphOutNeg{1},idxGraphOutNeg{2},idxGraphOutNeg{3});
	cmpsNeg = unique(segNeg.*maskNeg);
	cmpsNeg = cmpsNeg(cmpsNeg~=0);
	% reject patch if it doesn't have more than one object!
	if length(cmpsNeg)<2,
		if m.params.debug, disp('Negative patch doesn''t have more than one object!'), end
		continue
	end
	connNeg = MakeConnLabel(segNeg.*maskNeg,m.params.nhood);
	


	% Pick big patch for positive examples
	imPickPos = randsample(length(im),1);
	segPickPos = randsample(length(seg{imPickPos}),1);

% 	for k=1:3,
% 		idx{k} = (m.leftBorder(k)+half_big_output_size(k)+1):(imSz{imPickPos}-m.rightBorder(k)-half_big_output_size(k));
% 	end
% 	wt = mask{imPickPos}{segPickPos}(idx{1},idx{2},idx{3});
% 	if ~sum(wt(:)), continue, end
% 	[index_bigpos(1) index_bigpos(2) index_bigpos(3)] = ind2sub(imSzValid{k},randsample(wt(:),1,1));
	for k=1:3,
		index_bigpos(k) = randi(imSz{imPickPos}(k)-m.totalBorder(k)-big_output_size(k)+1,1);
	end
	index_bigpos(4) = imPickPos;
	for k=1:3,
		idxGraphOutPos{k} = (index_bigpos(k)+m.leftBorder(k)-1)+(1:big_output_size(k));
	end
	segPos = single(connectedComponents( ...
				MakeConnLabel( ...
					seg{imPickPos}{segPickPos}(idxGraphOutPos{1},idxGraphOutPos{2},idxGraphOutPos{3}), ...
					m.params.nhood),m.params.nhood));
	maskPos = mask{imPickPos}{segPickPos}(idxGraphOutPos{1},idxGraphOutPos{2},idxGraphOutPos{3});
	cmpsPos = unique(segPos.*maskPos);
	cmpsPos = cmpsPos(cmpsPos~=0);
	% reject patch if it doesn't have even one object!
	if length(cmpsPos)<1,
		if m.params.debug,disp('Positive patch doesn''t have even one object!'), end
		continue
	end
	connPos = MakeConnLabel(segPos.*maskPos,m.params.nhood);



	%% Run the fwd pass & get the output ---------------------------------------------------------
% fprintf('Running big fwd pass on gpu...'),tic
	cns('set',{m.layer_map.bigpos_minibatch_index,'val',index_bigpos-1},{m.layer_map.bigneg_minibatch_index,'val',index_bigneg-1});
	[inoutEstPos,inoutEstNeg] = cns('step',m.step_map.bigpos_fwd(1),m.step_map.bigneg_fwd(2),{m.layer_map.bigpos_output,'val'},{m.layer_map.bigneg_output,'val'});
	connEstPos = inout2conn(permute(inoutEstPos,[2 3 4 1]),m.params.nhood);
	connEstNeg = inout2conn(permute(inoutEstNeg,[2 3 4 1]),m.params.nhood);
% fprintf(' done. '),toc
% fprintf('everything in between...'),tic



	%% Pick positive and negative examples -------------------------------------------------------
	index(:)=0; mblabel(:)=0; mbmask(:)=0;
	index(:,1,1:m.params.minibatch_size) = imPickNeg;
	index(:,1,m.params.minibatch_size+(1:m.params.minibatch_size)) = imPickPos;
	pt1 = zeros(3,1); pt2 = zeros(3,1);
	for iex = 1:m.params.minibatch_size,
		tryAgain = 1; nTries = 0;
		while tryAgain,
			cmpPick = randsample(cmpsNeg,1);
			[pt1(1) pt1(2) pt1(3)] = ind2sub(m.params.graph_size,randsample(find(maskNeg(:).*(segNeg(:)==cmpPick)),1));
			if sum(cmpsNeg~=cmpPick)~=1,
				cmpPick = randsample(cmpsNeg(cmpsNeg~=cmpPick),1);
			else,
				cmpPick = cmpsNeg(cmpsNeg~=cmpPick);
			end
			[pt2(1) pt2(2) pt2(3)] = ind2sub(m.params.graph_size,randsample(find(maskNeg(:).*(segNeg(:)==cmpPick)),1));
			if m.params.constrained_minimax,
				[minEdge,edgeWt] = maximinEdge(max(connEstNeg,connNeg-1e-5),m.params.nhood,pt1,pt2);
			else,
				[minEdge,edgeWt] = maximinEdge(connEstNeg,m.params.nhood,pt1,pt2);
			end
			tryAgain = isempty(minEdge);
			nTries = nTries + 1;
			if nTries >= 100, error, end
		end
		index(1:3,1,iex) = minEdge(1:3)+index_bigneg(1:3)-1;
		mblabel(minEdge(4),1,1,1,iex) = 0;
		mbmask(minEdge(4),1,1,1,iex) = 1;
	end
	for iex = 1:m.params.minibatch_size,
		tryAgain = 1; nTries = 0;
		while tryAgain,
			cmpPick = randsample(cmpsPos,1);
			[pt1(1) pt1(2) pt1(3)] = ind2sub(m.params.graph_size,randsample(find(maskPos(:).*(segPos(:)==cmpPick)),1));
			[pt2(1) pt2(2) pt2(3)] = ind2sub(m.params.graph_size,randsample(find(maskPos(:).*(segPos(:)==cmpPick)),1));
			if m.params.constrained_minimax,
				[minEdge,edgeWt] = maximinEdge(connEstPos.*connPos,m.params.nhood,pt1,pt2);
			else,
				[minEdge,edgeWt] = maximinEdge(connEstPos,m.params.nhood,pt1,pt2);
			end
			tryAgain = all(pt1==pt2);
			nTries = nTries + 1;
			if nTries >= 100, error, end
		end
		index(1:3,1,m.params.minibatch_size+iex) = minEdge(1:3)+index_bigpos(1:3)-1;
		mblabel(minEdge(4),1,1,1,m.params.minibatch_size+iex) = 1;
		mbmask(minEdge(4),1,1,1,m.params.minibatch_size+iex) = 1;
	end




	%% Do the training ---------------------------------------------------------------------------
% fprintf('done. '), toc
% fprintf('Running small training pass on gpu...'),tic
	cns('set',{m.layer_map.minibatch_index,'val',index-1},{m.layer_map.label,'val',mblabel},{m.layer_map.label,'mask',mbmask});
	[loss,classerr] = cns('step',m.step_map.fwd(1),m.step_map.gradient(end),{m.layer_map.output,'loss'},{m.layer_map.output,'classerr'});
% fprintf('done. '),toc

keyboard

	%% Record error statistics --------------------------------------------------------------------
	if m.params.debug >= 2,
		log_message(m,['DEBUG_MODE: loss: ' num2str(sum(loss(:)))])
	end
	m.stats.epoch_iter = rem(m.stats.iter,m.params.nIterPerEpoch)+1;
	m.stats.epoch_loss(m.stats.epoch_iter) = sum(loss(:).*mbmask(:))/sum(mbmask(:));
	m.stats.epoch_classerr(m.stats.epoch_iter) = sum(classerr(:).*mbmask(:))/sum(mbmask(:));
	m.stats.loss(m.stats.epoch) = mean(m.stats.epoch_loss(1:m.stats.epoch_iter));
	m.stats.classerr(m.stats.epoch) = mean(m.stats.epoch_classerr(1:m.stats.epoch_iter));

	%% Save current state ----------------------------------------------------------------------
	if (etime(clock,m.stats.last_backup)>m.params.backup_interval) || (m.stats.iter==length(m.stats.times)),
		log_message(m, ['Saving network state... Iter: ' num2str(m.stats.iter)]);
		m = cns('update',m);
		m.inputblock = {};
		for k=1:length(m.layers),
			switch m.layers{k}.type,
			case {'input', 'hidden', 'output', 'label'}
				if isfield(m.layers{k},'val'), m.layers{k}.val=0; end
				if isfield(m.layers{k},'sens'), m.layers{k}.sens=0; end
			end
		end
		save([m.params.save_string,'latest'],'m');
		m.stats.last_backup = clock;
	end

	%% Update counters --------------------------------------------------------------------------
	m.stats.iter = m.stats.iter+1;

	%% Compute test/train statistics ------------------------------------------------------------
	if ~rem(m.stats.iter,m.params.nIterPerEpoch*m.params.nEpochPerSave),
		%% save current state/statistics
		m = cns('update',m);
		m.inputblock = {};
		for k=1:length(m.layers),
			switch m.layers{k}.type,
			case {'input', 'hidden', 'output', 'label'}
				if isfield(m.layers{k},'val'), m.layers{k}.val=0; end
				if isfield(m.layers{k},'sens'), m.layers{k}.sens=0; end
			end
		end
		save([m.params.save_string,'epoch-' num2str(m.stats.epoch)],'m');
		save([m.params.save_string,'latest'],'m');
		%% new epoch
		log_message(m,['Epoch: ' num2str(m.stats.epoch) ', Iter: ' num2str(m.stats.iter) '; Classification error: ' num2str(m.stats.classerr(m.stats.epoch))]);
	end
	if ~rem(m.stats.iter,m.params.nIterPerEpoch),
		m.stats.epoch = m.stats.epoch+1;
	end

	%% Plot statistics ------------------------------------------------------------
	try,
		if ~rem(m.stats.iter,2e1),
			disp(['Loss(iter: ' num2str(m.stats.iter) ') ' num2str(m.stats.epoch_loss(m.stats.epoch_iter)) ', classerr: ' num2str(m.stats.epoch_classerr(m.stats.epoch_iter))])
			figure(1)
			subplot(121)
			plot(1:m.stats.epoch,m.stats.loss(1:m.stats.epoch))
			subplot(122)
			plot(1:m.stats.epoch,m.stats.classerr(1:m.stats.epoch))
			drawnow
		end
	catch,
	end

end

return


return
