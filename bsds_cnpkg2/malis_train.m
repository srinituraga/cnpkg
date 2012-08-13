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
m.totalBorder = cell2mat(m.layers{m.layer_map.input}.size(2:4)) - cell2mat(m.layers{m.layer_map.output}.size(2:4));
m.leftBorder = m.offset;
m.rightBorder = m.totalBorder - m.leftBorder;

% load data
data = load(m.data_info.training_files{1});
log_message(m,['Loaded data file...']);



% reformat 'im', apply mask, compute conn
transform_and_subsample_data;
m.inputblock = im;
big_output_size = cell2mat(m.layers{m.layer_map.big_output}.size(2:4));
total_offset = cell2mat(m.layers{m.layer_map.big_input}.size(2:4))-1;%m.offset + big_output_size - 1;
m.layers{m.layer_map.label}.mask = ones(cell2mat(m.layers{m.layer_map.output}.size),'single');
m.layers{m.layer_map.label}.val(1,1,1,1,1:m.params.minibatch_size) = 1;
m.layers{m.layer_map.label}.val(1,1,1,1,m.params.minibatch_size+(1:m.params.minibatch_size)) = 0;
fprintf('Initializing on gpu...'),tic
cns('init',m)
fprintf(' done. '),toc

% if epoch 0, save
if m.stats.epoch <= 1,
	m = cns('update',m);
	m.inputblock = {};
	save([m.params.save_string,'epoch-' num2str(m.stats.epoch)],'m');
	save([m.params.save_string,'latest'],'m');
	m.stats.last_backup = clock;
end


% done initializing!
log_message(m, ['initialization complete!']);



%% ------------------------Training
log_message(m, ['beginning training.. ']);
index_big = zeros(cell2mat(m.layers{m.layer_map.big_minibatch_index}.size));
index = zeros(cell2mat(m.layers{m.layer_map.minibatch_index}.size));
while(m.stats.iter < m.params.maxIter),

	%% Assemble a minibatch ---------------------------------------------------------------------
	%% select a nice juicy image and image patch
	tryAgain = 1; while tryAgain,
		imPick = randsample(length(im),1);
		segPick = randsample(length(seg{imPick}),1);

		for k = 1:3,
			index_big(k) = randi(imSz{imPick}(k)-total_offset(k),1,1);
		end
		index_big(4) = imPick;
		index(4,1,:) = imPick;

		for k = 1:3,
			idxOut{k} = index_big(k) + m.offset(k) + (1:big_output_size(k)) - 1;
		end

		segTrue = 5+connectedComponents( ...
						MakeConnLabel( ...
							seg{imPick}{segPick}(idxOut{1},idxOut{2},idxOut{3})+5, ...
								m.params.nhood), ...
								m.params.nhood);
		if m.params.constrained_minimax,
			segTrue = segTrue .* mask{imPick}{segPick}(idxOut{1},idxOut{2},idxOut{3});
		end
		segTrueNonZero = segTrue~=0; findSegTrueNonZero = find(segTrueNonZero);
		cmps = unique(segTrue(segTrueNonZero));
	tryAgain = length(cmps) < 2; end

	% run the fwd pass & get the output
% fprintf('Running big fwd pass on gpu...'),tic
	cns('set',{m.layer_map.big_minibatch_index,'val',index_big-1});
	inoutEst = cns('step',m.step_map.big_fwd(1),m.step_map.big_fwd(2),{m.layer_map.big_output,'val'});
	inoutEst = permute(inoutEst,[2 3 1]);
% fprintf(' done. '),toc
% fprintf('everything in between...'),tic

	% pick some pixel pairs
	for iex = 1:m.params.minibatch_size,
		% positive
		[pt1pos(iex,1),pt1pos(iex,2),pt1pos(iex,3)] = ind2sub(big_output_size,randsample(findSegTrueNonZero,1,1));
		[pt2pos(iex,1),pt2pos(iex,2),pt2pos(iex,3)] = ind2sub(big_output_size,randsample(find( ...
									(segTrue == segTrue(pt1pos(iex,1),pt1pos(iex,2),pt1pos(iex,3)))),1,1));
		% negative
		[pt1neg(iex,1),pt1neg(iex,2),pt1neg(iex,3)] = ind2sub(big_output_size,randsample(findSegTrueNonZero,1,1));
		[pt2neg(iex,1),pt2neg(iex,2),pt2neg(iex,3)] = ind2sub(big_output_size,randsample(find( ...
									((segTrue ~= segTrue(pt1neg(iex,1),pt1neg(iex,2),pt1neg(iex,3))) ...
									.*segTrueNonZero)),1,1));
	end
if (any(pt2neg(:)==0)), keyboard, end

	% find the minimax nodes
% fprintf('computing +ve maximin edge...'),tic
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
		index(k,:) = index(k,:) + index_big(k) - 1;
	end


	% do the training
	cns('set',{m.layer_map.minibatch_index,'val',index-1});
	[loss,classerr] = cns('step',m.step_map.fwd(1),m.step_map.gradient(end),{m.layer_map.output,'loss'},{m.layer_map.output,'classerr'});

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
		m.inputblock = {};
		save([m.params.save_string,'latest'],'m');
		m.stats.last_backup = clock;
	end

try,
if ~rem(m.stats.iter,2e1),
disp(['Loss(iter: ' num2str(m.stats.iter) ') ' num2str(mean(loss(:))) ', classerr: ' num2str(mean(classerr(:)))])
[i,o]=cns('get',{m.layer_map.big_input,'val'},{m.layer_map.big_output,'val'});
figure(2)
index2 = index(1:3,:)-repmat(index_big(1:3),1,m.params.minibatch_size*2);
subplot(221)
imagesc(permute(i(:,m.offset(1)+(1:big_output_size(1)),m.offset(2)+(1:big_output_size(2))),[2 3 1]))
hold on
plot(index2(2,1:m.params.minibatch_size), ...
		index2(1,1:m.params.minibatch_size),'r+', ...
	index2(2,m.params.minibatch_size+(1:m.params.minibatch_size)), ...
		index2(1,m.params.minibatch_size+(1:m.params.minibatch_size)),'b*');
hold off
subplot(222)
imagesc(permute(1-o,[2 3 1]),[0 1]),colormap gray
hold on
plot(index2(2,1:m.params.minibatch_size), ...
		index2(1,1:m.params.minibatch_size),'r+', ...
	index2(2,m.params.minibatch_size+(1:m.params.minibatch_size)), ...
		index2(1,m.params.minibatch_size+(1:m.params.minibatch_size)),'b*');
hold off
subplot(223)
imagesc(segTrue)
hold on
plot(index2(2,1:m.params.minibatch_size), ...
		index2(1,1:m.params.minibatch_size),'r+', ...
	index2(2,m.params.minibatch_size+(1:m.params.minibatch_size)), ...
		index2(1,m.params.minibatch_size+(1:m.params.minibatch_size)),'b*');
hold off
subplot(224)
segEst = watershed(imhmin(1-squeeze(o),.5,4),4);
imagesc(segEst)
hold on
plot(index2(2,1:m.params.minibatch_size), ...
		index2(1,1:m.params.minibatch_size),'r+', ...
	index2(2,m.params.minibatch_size+(1:m.params.minibatch_size)), ...
		index2(1,m.params.minibatch_size+(1:m.params.minibatch_size)),'b*');
hold off

figure(3)
subplot(121)
plot(1:m.stats.epoch,m.stats.loss(1:m.stats.epoch))
subplot(122)
plot(1:m.stats.epoch,m.stats.classerr(1:m.stats.epoch))
% figure(10)
% subplot(121)
% plot(m.stats.epoch_loss(1:m.stats.epoch_iter))
% subplot(122)
% plot(m.stats.epoch_classerr(1:m.stats.epoch_iter))
drawnow
end
catch,
end


	%% Update counters --------------------------------------------------------------------------
	m.stats.iter = m.stats.iter+1;

	%% Compute test/train statistics ------------------------------------------------------------
	if ~rem(m.stats.iter,m.params.nIterPerEpoch*m.params.nEpochPerSave),
		%% save current state/statistics
		m = cns('update',m);
		m.inputblock = {};
		save([m.params.save_string,'epoch-' num2str(m.stats.epoch)],'m');
		save([m.params.save_string,'latest'],'m');
		m.stats.last_backup = clock;
		%% new epoch
		log_message(m,['Epoch: ' num2str(m.stats.epoch) ', Iter: ' num2str(m.stats.iter) '; Classification error: ' num2str(m.stats.classerr(m.stats.epoch))]);
	end
	if ~rem(m.stats.iter,m.params.nIterPerEpoch),
		m.stats.epoch = m.stats.epoch+1;

		%% Reload data every so often --------------------------------------------------------------
		if ~rem(m.stats.epoch,m.params.nEpochPerDataBlock),
			transform_and_subsample_data;
			for mvIdx = 1:m.params.nDataBlock,
				cns('set',{0,'inputblock',mvIdx,im{mvIdx}});
			end
		end
	end

end


function transform_and_subsample_data
log_message(m,['Assembling dataBlock']);

for imIdx = 1:m.params.nDataBlock,

	% pick image and seg
	imPick = randsample(length(data.im),1);
	segPick = randsample(length(data.seg{imPick}),1);

	% pick a rescale factor
	rescale = m.params.dataBlockTransformRescale(1) + rand*diff(m.params.dataBlockTransformRescale);

	im{imIdx} = permute(imresize(squeeze(data.im{imPick}),rescale),[1 2 4 3]);
	im{imIdx} = min(1,max(0,im{imIdx}));
	seg{imIdx} = {}; mask{imIdx} = {}; bb{imIdx} = {};
	for segIdx = 1:length(data.seg{imPick}),
		seg{imIdx}{segIdx} = imresize(squeeze(data.seg{imPick}{segIdx}),rescale,'nearest');
		mask{imIdx}{segIdx} = imresize(squeeze(data.mask{imPick}{segIdx}),rescale,'nearest');
		bb{imIdx}{segIdx} = 1+floor((data.bb{imPick}{segIdx}-1)*rescale);
	end

	% crop
	for k=1:3,
		stidx = randi([bb{imIdx}{segPick}(k,1) ...
							max(bb{imIdx}{segPick}(k,1), ...
								bb{imIdx}{segPick}(k,2)-m.params.dataBlockSize(k))],1);
		idx{k} = stidx+(0:min(bb{imIdx}{segPick}(k,2),m.params.dataBlockSize(k))-1);
	end
	im{imIdx} = im{imIdx}(idx{1},idx{2},idx{3},:);
	if any(size2(im{imIdx},1:3) < m.params.dataBlockSize),
		im{imIdx}(m.params.dataBlockSize(1),m.params.dataBlockSize(2),m.params.dataBlockSize(3),:)=0;
	end
	for segIdx = 1:length(seg{imIdx}),
		seg{imIdx}{segIdx} = seg{imIdx}{segIdx}(idx{1},idx{2},idx{3});
		mask{imIdx}{segIdx} = mask{imIdx}{segIdx}(idx{1},idx{2},idx{3});
		if any(size2(seg{imIdx}{segIdx},1:3) < m.params.dataBlockSize),
			seg{imIdx}{segIdx}(m.params.dataBlockSize(1),m.params.dataBlockSize(2),m.params.dataBlockSize(3),:)=0;
			mask{imIdx}{segIdx}(m.params.dataBlockSize(1),m.params.dataBlockSize(2),m.params.dataBlockSize(3),:)=0;
		end
		for k=1:3,
			bb{imIdx}{segIdx}(k,:) = max(1,bb{imIdx}{segIdx}(k,:)-idx{k}(1)+1);
			bb{imIdx}{segIdx}(k,2) = min(bb{imIdx}{segIdx}(k,2),m.params.dataBlockSize(k));
		end
	end


	% transform

	flp1 = (rand>.5)&m.params.dataBlockTransformFlp(1);
	flp2 = (rand>.5)&m.params.dataBlockTransformFlp(2);
	flp3 = (rand>.5)&m.params.dataBlockTransformFlp(3);
	prmt = (rand>.5)&m.params.dataBlockTransformPrmt(1);

	if prmt,
		im{imIdx} = permute(im{imIdx},[2 1 3 4]);
		for segIdx = 1:length(seg{imIdx}),
			seg{imIdx}{segIdx} = permute(seg{imIdx}{segIdx},[2 1 3 4]);
			mask{imIdx}{segIdx} = permute(mask{imIdx}{segIdx},[2 1 3 4]);
			bb{imIdx}{segIdx} = bb{imIdx}{segIdx}([2 1 3],:);
		end
	end

	if flp1,
		im{imIdx} = flipdim(im{imIdx},1);
		for segIdx = 1:length(seg{imIdx}),
			seg{imIdx}{segIdx} = flipdim(seg{imIdx}{segIdx},1);
			mask{imIdx}{segIdx} = flipdim(mask{imIdx}{segIdx},1);
			oldbb = bb{imIdx}{segIdx};
			bb{imIdx}{segIdx}(1,1) = size(im{imIdx},1)-oldbb(1,2)+1;
			bb{imIdx}{segIdx}(1,2) = size(im{imIdx},1)-oldbb(1,1)+1;
		end
	end

	if flp2,
		im{imIdx} = flipdim(im{imIdx},2);
		for segIdx = 1:length(seg{imIdx}),
			seg{imIdx}{segIdx} = flipdim(seg{imIdx}{segIdx},2);
			mask{imIdx}{segIdx} = flipdim(mask{imIdx}{segIdx},2);
			oldbb = bb{imIdx}{segIdx};
			bb{imIdx}{segIdx}(2,1) = size(im{imIdx},2)-oldbb(2,2)+1;
			bb{imIdx}{segIdx}(2,2) = size(im{imIdx},2)-oldbb(2,1)+1;
		end
	end

	if flp3,
		im{imIdx} = flipdim(im{imIdx},3);
		for segIdx = 1:length(seg{imIdx}),
			seg{imIdx}{segIdx} = flipdim(seg{imIdx}{segIdx},3);
			mask{imIdx}{segIdx} = flipdim(mask{imIdx}{segIdx},3);
			oldbb = bb{imIdx}{segIdx};
			bb{imIdx}{segIdx}(3,1) = size(im{imIdx},3)-oldbb(3,2)+1;
			bb{imIdx}{segIdx}(3,2) = size(im{imIdx},3)-oldbb(3,1)+1;
		end
	end

	if isfield(m.params,'zero_pad') && m.params.zero_pad,
		aa = im{imIdx};
		sz = size2(aa,1:4);
		im{imIdx} = zeros(sz+[m.totalBorder 0],'single');
		im{imIdx}(m.leftBorder(1)+(1:sz(1)), ...
				m.leftBorder(2)+(1:sz(2)), ...
				m.leftBorder(3)+(1:sz(3)),:) ...
				= aa;
		imSz{imIdx} = size2(im{imIdx},1:3);

		for segIdx = 1:length(seg{imIdx}),
			bb{imIdx}{segIdx} = [1 1 1;size2(im{imIdx},[1 2 3])]';
			aa = seg{imIdx}{segIdx};
			sz = size2(aa,1:4);
			seg{imIdx}{segIdx} = zeros(sz+[m.totalBorder 0],'single');
			seg{imIdx}{segIdx}(m.leftBorder(1)+(1:sz(1)), ...
					m.leftBorder(2)+(1:sz(2)), ...
					m.leftBorder(3)+(1:sz(3)),:) ...
					= aa;

			aa = mask{imIdx}{segIdx};
			sz = size2(aa,1:4);
			mask{imIdx}{segIdx} = zeros(sz+[m.totalBorder 0],'single');
			mask{imIdx}{segIdx}(m.leftBorder(1)+(1:sz(1)), ...
					m.leftBorder(2)+(1:sz(2)), ...
					m.leftBorder(3)+(1:sz(3)),:) ...
					= aa;

		end

	end
	clear aa;

	im{imIdx} = permute(im{imIdx},[4 1 2 3]);

end
end

end
