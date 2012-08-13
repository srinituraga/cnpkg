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

if isfield(m.params,'zero_pad') && m.params.zero_pad,
	for k = 1:length(im),
		aa = im{k};
		for j = 1:4,
			sz(j) = size(aa,j);
		end
		im{k} = zeros(sz+[totalBorder 0],'single');
		im{k}(leftBorder(1)+(1:sz(1)), ...
				leftBorder(2)+(1:sz(2)), ...
				leftBorder(3)+(1:sz(3)),:) ...
				= aa;

		for kk = 1:length(seg{k}),
			aa = seg{k}{kk};
			for j = 1:4,
				sz(j) = size(aa,j);
			end
			seg{k}{kk} = zeros(sz+[totalBorder 0],'single');
			seg{k}{kk}(leftBorder(1)+(1:sz(1)), ...
					leftBorder(2)+(1:sz(2)), ...
					leftBorder(3)+(1:sz(3)),:) ...
					= aa;

			aa = mask{k}{kk};
			for j = 1:4,
				sz(j) = size(aa,j);
			end
			mask{k}{kk} = zeros(sz+[totalBorder 0],'single');
			mask{k}{kk}(leftBorder(1)+(1:sz(1)), ...
					leftBorder(2)+(1:sz(2)), ...
					leftBorder(3)+(1:sz(3)),:) ...
					= aa;

			aa = edge{k}{kk};
			for j = 1:4,
				sz(j) = size(aa,j);
			end
			edge{k}{kk} = zeros(sz+[totalBorder 0],'single');
			edge{k}{kk}(leftBorder(1)+(1:sz(1)), ...
					leftBorder(2)+(1:sz(2)), ...
					leftBorder(3)+(1:sz(3)),:) ...
					= aa;
		end

	end
end

% reformat 'im', apply mask, compute conn
for k = 1:length(im),
	im{k} = permute(im{k},[4 1 2 3]);
	imSz{k} = size2(im{k},2:4);
	imSzValid{k} = size2(im{k},2:4)-m.totalBorder-m.params.graph_size+1;
	% construct a weight for each big patch window
	% based on the numbers of positive and negative examples in that window
	% uses a binarized mask
	for kk = 1:length(seg{k}),
		myseg = seg{k}{kk}( ...
			m.leftBorder(1)+1:end-m.rightBorder(1), ...
			m.leftBorder(2)+1:end-m.rightBorder(2), ...
			m.leftBorder(3)+1:end-m.rightBorder(3));
		mymask = mask{k}{kk}( ...
			m.leftBorder(1)+1:end-m.rightBorder(1), ...
			m.leftBorder(2)+1:end-m.rightBorder(2), ...
			m.leftBorder(3)+1:end-m.rightBorder(3))>0;
		wtpos{k}{kk} = single(convn_fast((myseg>0).*mymask,ones(m.params.graph_size),'valid'));
		wtneg{k}{kk} = zeros(size(wtpos{k}{kk}),'single');
		cmps = unique(myseg);
		for j=cmps(cmps~=0)',
			wtneg{k}{kk} = wtneg{k}{kk}+(convn_fast((myseg==j).*mymask,ones(m.params.graph_size),'valid')>0);
		end
		wtneg{k}{kk} = max(0,wtneg{k}{kk}-1);
	end
end
m.layers{m.layer_map.bigpos_input}.inputblock = im;
m.layers{m.layer_map.bigneg_input}.inputblock = im;
m.layers{m.layer_map.input}.inputblock = im;
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
while(m.stats.iter < m.params.maxIter),

	%% Assemble a minibatch ---------------------------------------------------------------------
	%% select a nice juicy image and image patch

	% Pick big patch for positive examples
	imPickPos = randsample(length(im),1);
	segPickPos = randsample(length(seg{imPickPos}),1);

	index_bigpos(1:3) = sub2ind(imSzValid{imPickPos}, ...
							randsample(reshape(wtpos{imPickPos}{segPickPos},1,[]),1,1));
	index_bigpos(4) = imPickPos;
	index(4,1,1:m.params.minibatch_size) = imPickPos;

	for k = 1:3,
		idxposOut{k} = index_bigpos(k) + m.leftBorder(k) + (1:big_output_size(k)) - 1;
	end

	% Pick big patch for negative examples
	imPickNeg = randsample(length(im),1);
	segPickNeg = randsample(length(seg{imPickNeg}),1);

	index_bigneg(1:3) = sub2ind(imSzValid{imPickNeg}, ...
							randsample(reshape(wtneg{imPickNeg}{segPickNeg},1,[]),1,1));
	index_bigneg(4) = imPickNeg;
	index(4,1,m.params.minibatch_size+(1:m.params.minibatch_size)) = imPickNeg;

	for k = 1:3,
		idxnegOut{k} = index_bigneg(k) + m.leftBorder(k) + (1:big_output_size(k)) - 1;
	end



	%% Run the fwd pass & get the output ---------------------------------------------------------
% fprintf('Running big fwd pass on gpu...'),tic
	cns('set',{m.layer_map.bigpos_minibatch_index,'val',index_bigpos-1},{m.layer_map.bigneg_minibatch_index,'val',index_bigneg-1});
	[connPosEst,connNegEst] = cns('step',m.step_map.bigpos_fwd(1),m.step_map.bigneg_fwd(2),{m.layer_map.bigpos_output,'val'},{m.layer_map.bigneg_output,'val'});
	connPosEst = permute(connPosEst,[2 3 4 1]);
	connNegEst = permute(connNegEst,[2 3 4 1]);
% fprintf(' done. '),toc
% fprintf('everything in between...'),tic



	%% Pick positive and negative examples -------------------------------------------------------
	segPosTrue = 5+connectedComponents( ...
					MakeConnLabel( ...
						seg{imPickPos}{segPickPos}(idxposOut{1},idxposOut{2},idxposOut{3})+5, ...
							m.params.nhood), ...
							m.params.nhood);
	segPosTrue = segPosTrue .* mask{imPickPos}{segPickPos}(idxposOut{1},idxposOut{2},idxposOut{3});
	segPosTrueNonZero = segPosTrue~=0; findSegPosTrueNonZero = find(segPosTrueNonZero);
	cmpsPos = unique(segTrue(segPosTrueNonZero));
	% pick some pixel pairs
	% positive
	for iex = 1:m.params.minibatch_size,
		[pt1pos(iex,1),pt1pos(iex,2),pt1pos(iex,3)] = ind2sub(big_output_size,randsample(findSegPosTrueNonZero,1,1));
		[pt2pos(iex,1),pt2pos(iex,2),pt2pos(iex,3)] = ind2sub(big_output_size,randsample(find( ...
									(segPosTrue == segPosTrue(pt1pos(iex,1),pt1pos(iex,2),pt1pos(iex,3)))),1,1));
	end

	segNegTrue = 5+connectedComponents( ...
					MakeConnLabel( ...
						seg{imPickNeg}{segPickNeg}(idxNegOut{1},idxNegOut{2},idxNegOut{3})+5, ...
							m.params.nhood), ...
							m.params.nhood);
	segNegTrue = segNegTrue .* mask{imPickNeg}{segPickNeg}(idxNegOut{1},idxNegOut{2},idxNegOut{3});
	segNegTrueNonZero = segNegTrue~=0; findSegNegTrueNonZero = find(segNegTrueNonZero);
	cmpsNeg = unique(segTrue(segNegTrueNonZero));
	% negative
	for iex = 1:m.params.minibatch_size,
		[pt1neg(iex,1),pt1neg(iex,2),pt1neg(iex,3)] = ind2sub(big_output_size,randsample(findSegNegTrueNonZero,1,1));
		[pt2neg(iex,1),pt2neg(iex,2),pt2neg(iex,3)] = ind2sub(big_output_size,randsample(find( ...
									((segNegTrue ~= segNegTrue(pt1neg(iex,1),pt1neg(iex,2),pt1neg(iex,3))) ...
									.*segNegTrueNonZero)),1,1));
	end
if (any(pt2neg(:)==0)), keyboard, end

	%% Find the minimax nodes
% fprintf('computing +ve maximin edge...'),tic
	mbmask = zeros(cell2mat(m.layers{m.layer_map.output}.size),'single');
	mblabel = zeros(cell2mat(m.layers{m.layer_map.output}.size),'single');
	inoutEstPos = inoutEst;
	if m.params.constrained_minimax,
		inoutEstPos(~segTrueNonZero) = -1;
	end
	minEdge = maximinEdge(inout2conn(inoutEstPos,m.params.nhood),m.params.nhood,pt1pos',pt2pos');
	minEdge = minEdge';
% fprintf('done. '),toc
	left = inoutEstPos(sub2ind(big_output_size, ...
								minEdge(:,1), ...
								minEdge(:,2), ...
								minEdge(:,3))) ...
			< inoutEstPos(sub2ind(big_output_size, ...
								minEdge(:,1)+m.params.nhood(minEdge(:,4),1), ...
								minEdge(:,2)+m.params.nhood(minEdge(:,4),2), ...
								minEdge(:,3)+m.params.nhood(minEdge(:,4),3)));
	for k = 1:3,
		index(k,1,find(left)) = minEdge(find(left),k);
		index(k,1,find(~left)) = minEdge(find(~left),k)+m.params.nhood(minEdge(find(~left),4),k);
	end
% fprintf('computing -ve maximin edge...'),tic
	inoutEstNeg = inoutEst;
	if m.params.constrained_minimax,
		inoutEstNeg(segTrueNonZero) = 2;
	end
	minEdge = maximinEdge(inout2conn(inoutEstNeg,m.params.nhood),m.params.nhood,pt1neg',pt2neg');
	minEdge = minEdge';
% fprintf('done. '),toc
	left = inoutEstNeg(sub2ind(big_output_size, ...
								minEdge(:,1), ...
								minEdge(:,2), ...
								minEdge(:,3))) ...
			< inoutEstNeg(sub2ind(big_output_size, ...
								minEdge(:,1)+m.params.nhood(minEdge(:,4),1), ...
								minEdge(:,2)+m.params.nhood(minEdge(:,4),2), ...
								minEdge(:,3)+m.params.nhood(minEdge(:,4),3)));
	for k = 1:3,
		index(k,1,m.params.minibatch_size+find(left)) = minEdge(find(left),k);
		index(k,1,m.params.minibatch_size+find(~left)) = minEdge(find(~left),k)+m.params.nhood(minEdge(find(~left),4),k);
	end

	for k = 1:3,
		index(k,1:m.params.minibatch_size) = index(k,1:m.params.minibatch_size) + index_bigpos(k) - 1;
		index(k,m.params.minibatch_size+(1:m.params.minibatch_size)) = index(k,m.params.minibatch_size+(1:m.params.minibatch_size)) + index_bigneg(k) - 1;
	end


	%% Do the training ---------------------------------------------------------------------------
% fprintf('done. '), toc
% fprintf('Running small training pass on gpu...'),tic
	cns('set',{m.layer_map.minibatch_index,'val',index-1},{m.layer_map.label,'val',mblabel},{m.layer_map.label,'mask',mbmask});
	[loss,classerr] = cns('step',m.step_map.fwd(1),m.step_map.gradient,{m.layer_map.output,'loss'},{m.layer_map.output,'classerr'});
% fprintf('done. '),toc


	%% Record error statistics --------------------------------------------------------------------
	if m.params.debug >= 2,
		log_message(m,['DEBUG_MODE: loss: ' num2str(sum(loss(:)))])
	end
	m.stats.epoch_iter = rem(m.stats.iter,m.params.nIterPerEpoch)+1;
	m.stats.epoch_loss(m.stats.epoch_iter) = mean(loss(:));
	m.stats.epoch_classerr(m.stats.epoch_iter) = mean(classerr(:));
	m.stats.loss(m.stats.epoch) = mean(m.stats.epoch_loss(1:m.stats.epoch_iter));
	m.stats.classerr(m.stats.epoch) = mean(m.stats.epoch_classerr(1:m.stats.epoch_iter));

	%% Save current state ----------------------------------------------------------------------
	if (etime(clock,m.stats.last_backup)>m.params.backup_interval) || (m.stats.iter==length(m.stats.times)),
		log_message(m, ['Saving network state... Iter: ' num2str(m.stats.iter)]);
		m = cns('update',m);
		m.layers{m.layer_map.big_input}.inputblock = {};
		m.layers{m.layer_map.input}.inputblock = {};
		save([m.params.save_string,'latest'],'m');
		m.stats.last_backup = clock;
	end

	%% Update counters --------------------------------------------------------------------------
	m.stats.iter = m.stats.iter+1;

	%% Compute test/train statistics ------------------------------------------------------------
	if ~rem(m.stats.iter,m.params.nIterPerEpoch*m.params.nEpochPerSave),
		%% save current state/statistics
		m = cns('update',m);
		m.layers{m.layer_map.big_input}.inputblock = [];
		m.layers{m.layer_map.input}.inputblock = [];
		save([m.params.save_string,'epoch-' num2str(m.stats.epoch)],'m');
		%% new epoch
		log_message(m,['Epoch: ' num2str(m.stats.epoch) ', Iter: ' num2str(m.stats.iter) '; Classification error: ' num2str(m.stats.classerr(m.stats.epoch))]);
	end
	if ~rem(m.stats.iter,m.params.nIterPerEpoch),
		m.stats.epoch = m.stats.epoch+1;
	end

	%% Plot statistics ------------------------------------------------------------
	try,
		if ~rem(m.stats.iter,2e1),
			disp(['Loss(iter: ' num2str(m.stats.iter) ') ' num2str(mean(loss(:))) ', classerr: ' num2str(mean(classerr(:)))])
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
