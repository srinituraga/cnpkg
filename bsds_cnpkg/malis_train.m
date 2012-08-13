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

% load data
load(m.data_info.training_files{1});
log_message(m,['Loaded data file...']);
% idx = randsample(length(im),2,1);
% im = im(idx); seg = seg(idx); mask = mask(idx);

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
if ~isfield(m.stats,'iter') || m.stats.iter < 1,
	m.stats.iter = 1;
end
if ~isfield(m.stats,'epoch') || m.stats.epoch < 1,
	m.stats.epoch = 1;
end
log_message(m, ['initializing stats']);

% construct cns model for gpu
m = malis_mapdim_layers(m);

% reformat 'im', apply mask, compute conn
for k = 1:length(im),
	im{k} = permute(im{k},[4 1 2 3]);
	imSz{k} = size2(im{k},2:4);
% 	for j = 1:length(seg{k}),
% 		seg{k}{j} = (5+seg{k}{j}).*mask{k}{j};
% 	end
end
m.layers{m.layer_map.big_input}.inputblock = im;
m.layers{m.layer_map.input}.inputblock = im;
offset = m.layers{m.layer_map.label}.offset;
big_output_size = cell2mat(m.layers{m.layer_map.big_output}.size(2:4));
total_offset = offset + big_output_size - 1;
m.layers{m.layer_map.label}.mask = ones(cell2mat(m.layers{m.layer_map.output}.size),'single');
m.layers{m.layer_map.label}.val(1,1,1,1,1:m.params.minibatch_size) = 1;
m.layers{m.layer_map.label}.val(1,1,1,1,m.params.minibatch_size+(1:m.params.minibatch_size)) = 0;
fprintf('Initializing on gpu...'),tic
cns('init',m,'gpu','mean')
fprintf(' done. '),toc


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
			idxOut{k} = index_big(k) + offset(k) + (1:big_output_size(k)) - 1;
		end

		segTrue = 5+connectedComponents( ...
						MakeConnLabel( ...
							seg{imPick}{segPick}(idxOut{1},idxOut{2},idxOut{3})+5, ...
								m.params.nhood), ...
								m.params.nhood);
		segTrue = segTrue .* mask{imPick}{segPick}(idxOut{1},idxOut{2},idxOut{3});
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
% segTrue(sub2ind(big_output_size,index(1,:),index(2,:),index(3,:)))
% [segTrue(sub2ind(big_output_size,pt1pos(:,1),pt1pos(:,2),pt1pos(:,3)))';
% segTrue(sub2ind(big_output_size,pt2pos(:,1),pt2pos(:,2),pt2pos(:,3)))']
% [segTrue(sub2ind(big_output_size,pt1neg(:,1),pt1neg(:,2),pt1neg(:,3)))';
% segTrue(sub2ind(big_output_size,pt2neg(:,1),pt2neg(:,2),pt2neg(:,3)))']
% keyboard
	for k = 1:3,
		index(k,:) = index(k,:) + index_big(k) - 1;
	end
% mask{imPick}{segPick}(sub2ind(size2(seg{imPick}{segPick},1:3),index(1,:)+offset(1),index(2,:)+offset(2),index(3,:)+offset(3)))


	% do the training
% fprintf('done. '), toc
% fprintf('Running small training pass on gpu...'),tic
%%%if m.stats.iter == 1,
	cns('set',{m.layer_map.minibatch_index,'val',index-1});
%%%	x{1} = zeros(cell2mat(m.layers{m.layer_map.input}.size),'single');
%%%	for k = 1:size(x{1},5),
%%%		x{1}(:,:,:,:,k) = im{index(4,1,k)}(:,index(1,1,k)+(0:offset(1)*2),index(2,1,k)+(0:offset(2)*2),index(3,1,k)+(0:offset(3)*2));
%%%	end
%%%end
%%%	for l=1:length(m.layer_map.weight),
%%%		W{l} = cns('get',m.layer_map.weight(l),'val');
%%%		b{l} = cns('get',m.layer_map.bias(l),'val');
%%%	end
%%%	f  = inline('1./(1+exp(-x))');
%%%	df = inline('x.*(1-x)');
%%%	lab = cns('get',m.layer_map.output,'label');
%%%	lab(:)'
%%%
	[loss,classerr] = cns('step',m.step_map.fwd(1),m.step_map.gradient,{m.layer_map.output,'loss'},{m.layer_map.output,'classerr'});
% fprintf('done. '),toc

%%%	xg{1} = cns('get',{m.layer_map.input,'val'});
%%%	xerr(1) = max(abs(xg{1}(:)-x{1}(:)));
%%%	for l = 1:m.params.num_layers-1,
%%%		x{l+1} = zeros(cell2mat(m.layers{m.layer_map.hidden(l)}.size),'single');
%%%		for j = 1:size(W{l},5),
%%%			x{l+1}(j,:,:,:) = b{l}(j);
%%%			for k = 1:size(W{l},1),
%%%				x{l+1}(j,:,:,:) = x{l+1}(j,:,:,:,:) + convn(x{l}(k,:,:,:,:),W{l}(k,:,:,:,j),'valid');
%%%			end
%%%		end
%%%		x{l+1} = f(x{l+1});
%%%		xg{l+1} = cns('get',{m.layer_map.hidden(l),'val'});
%%%		xerr(l+1) = max(abs(xg{l+1}(:)-x{l+1}(:)));
%%%	end
%%%	x{m.params.num_layers+1} = zeros(cell2mat(m.layers{m.layer_map.output}.size),'single');
%%%	for j = 1:size(W{end},5),
%%%		x{end}(j,:,:,:) = b{end}(j);
%%%		for k = 1:size(W{end},1),
%%%			x{end}(j,:,:,:) = x{end}(j,:,:,:,:) + convn(x{end-1}(k,:,:,:,:),W{end}(k,:,:,:,j),'valid');
%%%		end
%%%	end
%%%	x{end} = f(x{end});
%%%	xg{m.params.num_layers+1} = cns('get',{m.layer_map.output,'val'});
%%%	xerr(m.params.num_layers+1) = max(abs(xg{end}(:)-x{end}(:)));
%%%
%%%	s{m.params.num_layers+1} = (lab-x{end}).*df(x{end});
%%%	sg{m.params.num_layers+1} = cns('get',{m.layer_map.output,'sens'});
%%%	serr(m.params.num_layers+1) = max(abs(sg{end}(:)-s{end}(:)));
%%%	for l = m.params.num_layers:-1:2,
%%%		s{l} = zeros(cell2mat(m.layers{m.layer_map.hidden(l-1)}.size),'single');
%%%		for k = 1:size(W{l},1),
%%%			for j = 1:size(W{l},5),
%%%				s{l}(k,:,:,:) = s{l}(k,:,:,:,:) + convn(s{l+1}(j,:,:,:,:),flipdims(W{l}(k,:,:,:,j)),'full');
%%%			end
%%%		end
%%%		s{l} = s{l}.*df(x{l});
%%%		sg{l} = cns('get',{m.layer_map.hidden(l-1),'sens'});
%%%		serr(l) = max(abs(sg{l}(:)-s{l}(:)));
%%%	end
%%%
%%%	for l = 1:m.params.num_layers,
%%%		for j = 1:size(W{l},5),
%%%			for k = 1:size(W{l},1),
%%%				dW{l}(k,1:3,1:3,1,j) = flipdims(xcorrn(x{l}(k,:,:,:,:),s{l+1}(j,:,:,:,:),'valid'));
%%%			end
%%%		end
%%%		db{l} = sum(s{l+1}(:,:),2);
%%%		dWg{l} = cns('get',m.layer_map.weight(l),'dval');
%%%		dWerr(l) = max(abs(dWg{l}(:)-dW{l}(:)));
%%%		dbg{l} = cns('get',m.layer_map.bias(l),'dval');
%%%		dberr(l) = max(abs(dbg{l}(:)-db{l}(:)));
%%%	end
%%%
%%%	xerr,serr,dWerr,dberr
%%%%keyboard

	if m.params.debug >= 2,
		log_message(m,['DEBUG_MODE: loss: ' num2str(sum(loss(:)))])
	end
	m.stats.loss(m.stats.iter) = mean(loss(:));
	m.stats.classerr(m.stats.iter) = mean(classerr(:));

	%% Save current state ----------------------------------------------------------------------
	if (etime(clock,m.stats.last_backup)>m.params.backup_interval) || (m.stats.iter==length(m.stats.times)),
		log_message(m, ['Saving network state... Iter: ' num2str(m.stats.iter)]);
		m = cns('update',m);
		m.layers{m.layer_map.big_input}.inputblock = {};
		m.layers{m.layer_map.input}.inputblock = {};
		save([m.params.save_string,'latest'],'m');
		m.stats.last_backup = clock;
	end

try,
if ~rem(m.stats.iter,2e1),
disp(['Loss(iter: ' num2str(m.stats.iter) ') ' num2str(mean(loss(:))) ', classerr: ' num2str(mean(classerr(:)))])
[i,o]=cns('get',{m.layer_map.big_input,'val'},{m.layer_map.big_output,'val'});
figure(2)
subplot(311),imagesc(permute(i(:,offset(1)+1:end-offset(1),offset(2)+1:end-offset(2)),[2 3 1])),subplot(312),imagesc(permute(o,[2 3 1]),[0 1]),colormap gray,subplot(313),imagesc(segTrue)
figure(3)
plot(smooth(m.stats.classerr(1:m.stats.iter),1e3))
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
		m.layers{m.layer_map.big_input}.inputblock = [];
		m.layers{m.layer_map.input}.inputblock = [];
		save([m.params.save_string,'epoch-' num2str(m.stats.epoch)],'m');
		%% new epoch
		log_message(m,['Epoch: ' num2str(m.stats.epoch) ', Iter: ' num2str(m.stats.iter) '; Classification error: ' num2str(mean(m.stats.classerr(m.stats.iter-(m.params.nIterPerEpoch*m.params.nEpochPerSave-1):m.stats.iter)))]);
		m.stats.epoch = m.stats.epoch+1;
	end


end

return


return
