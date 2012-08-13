function m = cnpkg_mknet;

%%% Data files %%%
% specify data files (can add more to these structs at a later time too)
% training_set = load('~sturaga/net_sets/cnpkg/BSD/train_color_boundary_mean_smaller');
training_set.label{1} = single(rand(50,60,70)<.1);
training_set.input{1} = convn(training_set.label{1},ones(5,5,5)/5^3,'same');
training_set.input2label{1} = 1;
training_set.mask{1} = ones(size(training_set.label{1}),'single');

m.params.maxIter=5e6;
m.params.nIterPerEpoch = 1e4;
m.params.minibatch_size = 1;


%%% NETWORK ARCHITECTURE %%%
m.params.output_size=[1 1 1];
m.params.input_units=3;
m.params.output_units = 1;
m.params.num_layers=3;

% structure of each layer
m.params.layer{1}.nHid = {5};
m.params.layer{1}.scales{1} = [1 1 1];

m.params.layer{2}.nHid = {9};
m.params.layer{2}.scales{1} = [1 1 1];

% m.params.layer{1}.nHid = {3, 3, 3};
% m.params.layer{1}.scales{1} = [1 1 1];
% m.params.layer{1}.scales{2} = [3 3 1];
% m.params.layer{1}.scales{3} = [5 5 1];
% 
% m.params.layer{2}.nHid = {2, 2, 2};
% m.params.layer{2}.scales{1} = [1 1 1];
% m.params.layer{2}.scales{2} = [3 3 1];
% m.params.layer{2}.scales{3} = [5 5 1];

m.params.layer{m.params.num_layers}.scales{1} = [1 1 1];

m.params.globaleta = 1e-1;
for l = 1:m.params.num_layers,
	for s = 1:length(m.params.layer{l}.scales),
		m.params.layer{l}.patchSz{s} = [[1 1]*7 1];
		m.params.layer{l}.etaW{s} = 1;
		m.params.layer{l}.etaB{s} = 1;
		m.params.layer{l}.initW{s} = 5e-0;
		m.params.layer{l}.initB{s} = 5e-0;
	end
end
for s = 1:length(m.params.layer{end}.scales),
	m.params.layer{end}.etaW{s} = 1e-1;
	m.params.layer{end}.etaB{s} = 1e-1;
end
clear s

% set up network based on these parameters
% initialize random number generators
rand('state',0);
randn('state',0);

% build the gpu cns model structure
m = cnpkg2_buildmodel(m);
m = cnpkg2_mapdim_layers_bkwd(m,m.params.output_size,m.params.minibatch_size*2);

% initialize random number generators
rand('state',0);
randn('state',0);

m.stats.iter = 0;

% construct cns model for gpu
offset = m.offset;
totalBorder = 2*offset + cell2mat(m.layers{m.layer_map.output}.size(2:4)) - 1;
halfTotalBorder = ceil(totalBorder/2);

nInput = length(training_set.input);
nLabel = length(training_set.label);
m.layers{m.layer_map.input}.inputblock = training_set.input;
m.layers{m.layer_map.label}.labelblock = training_set.label;
m.layers{m.layer_map.label}.maskblock = training_set.mask;

m.layers{m.layer_map.minibatch_index}.size = {5,m.params.nIterPerEpoch,m.params.minibatch_size*2};
index = zeros(cell2mat(m.layers{m.layer_map.minibatch_index}.size));
m.layers{m.layer_map.minibatch_index}.val = index;

% initialize gpu
fprintf('Initializing on gpu...');tic
cns('init',m)
fprintf(' done. ');toc




%% ------------------------Training
while(m.stats.iter < m.params.maxIter),

	%% Assemble minibatch indices ---------------------------------------------------------------
% if m.stats.iter==0,
fprintf('Assembling minibatch indices...');tic
	index(4,:,:) = randi(nInput,[1 m.params.nIterPerEpoch m.params.minibatch_size*2]);

	% loop over images
	for k = 1:nInput,
		% pick a label image corresponding to this input image
		nLabel = length(training_set.input2label{k});
		idxLabel = find(squeeze(index(4,:,:))==k);
		index(5,idxLabel) = randi(nLabel,[1 length(idxLabel)]);

		for j = 1:nLabel,
			InputLabel = squeeze((index(4,:,:)==k) & (index(5,:,:)==j));
			pos = training_set.label{training_set.input2label{k}}(:, ...
										halfTotalBorder(1)+1:end-halfTotalBorder(1), ...
										halfTotalBorder(2)+1:end-halfTotalBorder(2), ...
										halfTotalBorder(3)+1:end-halfTotalBorder(3)) == 1;

			% positive examples
			idxInpLab = InputLabel;
			idxInpLab(:,m.params.minibatch_size+(1:m.params.minibatch_size)) = 0;
			idxInpLab = find(idxInpLab);
			idxPos = randsample(find(pos),length(idxInpLab),1);
			[junk,index(1,idxInpLab),index(2,idxInpLab),index(3,idxInpLab)] = ind2sub( ...
							[size(pos,1) size(pos,2) size(pos,3) size(pos,4)],idxPos);
			% negative examples
			idxInpLab = InputLabel;
			idxInpLab(:,(1:m.params.minibatch_size)) = 0;
			idxInpLab = find(idxInpLab);
			idxNeg = randsample(find(~pos),length(idxInpLab),1);
			[junk,index(1,idxInpLab),index(2,idxInpLab),index(3,idxInpLab)] = ind2sub( ...
							[size(pos,1) size(pos,2) size(pos,3) size(pos,4)],idxNeg);
		end

		index(5,idxLabel) = training_set.input2label{k}(index(5,idxLabel));
	end
fprintf('done. '); toc
index = repmat(index(:,1,:),[1 m.params.nIterPerEpoch 1]);
% end


	%% Train ------------------------------------------------------------------------------------
fprintf('Training on gpu...');tic
	cns('set',{0,'iter_no',0},{m.layer_map.minibatch_index,'val',index-1});
	[loss,classerr] = cns('run',m.params.nIterPerEpoch,{m.layer_map.output,'loss'},{m.layer_map.output,'classerr'});
fprintf('done. ');toc
	m.stats.loss(m.stats.iter+(1:m.params.nIterPerEpoch)) = mean(loss(:,:),2);
	m.stats.classerr(m.stats.iter+(1:m.params.nIterPerEpoch)) = mean(classerr(:,:),2);

	%% Compare with matlab implementation =================================
	x{1} = zeros(cell2mat(m.layers{m.layer_map.input}.size),'single');
	for k = 1:size(x{1},5),
		x{1}(:,:,:,:,k) = training_set.input{index(4,1,k)}(:,index(1,1,k)+(0:offset(1)*2+m.params.output_size(1)-1),index(2,1,k)+(0:offset(2)*2+m.params.output_size(2)-1),index(3,1,k)+(0:offset(3)*2+m.params.output_size(3)-1));
		lab(1,:,:,1,k) = training_set.label{index(4,1,k)}(:,index(1,1,k)+offset(1)+(0:m.params.output_size(1)-1),index(2,1,k)+offset(2)+(0:m.params.output_size(2)-1),index(3,1,k)+offset(3)+(0:m.params.output_size(3)-1));
	end
	for l=1:length(m.layer_map.weight),
		W{l} = cns('get',{m.layer_map.weight{l}{1},'val'});
		b{l} = cns('get',{m.layer_map.bias{l}{1},'val'});
	end
	f  = inline('1./(1+exp(-x))');
	df = inline('x.*(1-x)');
	labg = cns('get',{m.layer_map.label,'val'});
	[lab(:)';labg(:)']

	xg{1} = cns('get',{m.layer_map.input,'val'});
	xerr(1) = max(abs(xg{1}(:)-x{1}(:)));
	for l = 1:m.params.num_layers-1,
		x{l+1} = zeros(cell2mat(m.layers{m.layer_map.hidden{l}(1)}.size),'single');
		for j = 1:size(W{l},5),
			x{l+1}(j,:,:,:) = b{l}(j);
			for k = 1:size(W{l},1),
				x{l+1}(j,:,:,:) = x{l+1}(j,:,:,:,:) + convn(x{l}(k,:,:,:,:),W{l}(k,:,:,:,j),'valid');
			end
		end
		x{l+1} = f(x{l+1});
		xg{l+1} = cns('get',{m.layer_map.hidden{l}(1),'val'});
		xerr(l+1) = max(abs(xg{l+1}(:)-x{l+1}(:)));
	end
	x{m.params.num_layers+1} = zeros(cell2mat(m.layers{m.layer_map.output}.size),'single');
	for j = 1:size(W{end},5),
		x{end}(j,:,:,:) = b{end}(j);
		for k = 1:size(W{end},1),
			x{end}(j,:,:,:) = x{end}(j,:,:,:,:) + convn(x{end-1}(k,:,:,:,:),W{end}(k,:,:,:,j),'valid');
		end
	end
	x{end} = f(x{end});
	xg{m.params.num_layers+1} = cns('get',{m.layer_map.output,'val'});
	xerr(m.params.num_layers+1) = max(abs(xg{end}(:)-x{end}(:)));

	s{m.params.num_layers+1} = (lab-x{end}).*df(x{end});
	sg{m.params.num_layers+1} = cns('get',{m.layer_map.output,'sens'});
	serr(m.params.num_layers+1) = max(abs(sg{end}(:)-s{end}(:)));
	for l = m.params.num_layers:-1:2,
		s{l} = zeros(cell2mat(m.layers{m.layer_map.hidden{l-1}(1)}.size),'single');
		for k = 1:size(W{l},1),
			for j = 1:size(W{l},5),
				s{l}(k,:,:,:) = s{l}(k,:,:,:,:) + convn(s{l+1}(j,:,:,:,:),flipdims(W{l}(k,:,:,:,j)),'full');
			end
		end
		s{l} = s{l}.*df(x{l});
		sg{l} = cns('get',{m.layer_map.hidden{l-1}(1),'sens'});
		serr(l) = max(abs(sg{l}(:)-s{l}(:)));
	end

	for l = 1:m.params.num_layers,
		for j = 1:size(W{l},5),
			for k = 1:size(W{l},1),
				dW{l}(k,:,:,1,j) = flipdims(xcorrn(x{l}(k,:,:,:,:),s{l+1}(j,:,:,:,:),'valid'));
			end
		end
		db{l} = sum(s{l+1}(:,:),2);
		dWg{l} = cns('get',m.layer_map.weight{l}{1},'dval');
		dWerr(l) = max(abs(dWg{l}(:)-dW{l}(:)));
		dbg{l} = cns('get',m.layer_map.bias{l}{1},'dval');
		dberr(l) = max(abs(dbg{l}(:)-db{l}(:)));
	end

	xerr,serr,dWerr,dberr

		m.stats.iter = m.stats.iter + m.params.nIterPerEpoch;
		try,
			figure(1)
			plot(smooth(m.stats.loss(1:m.stats.iter),m.params.nIterPerEpoch))
			%plot(m.stats.loss(1:m.stats.iter))
			drawnow
		catch,
			warning('unable to plot')
		end

		try,
			figure(2)
			ii = cns('get',{m.layer_map.input,'val'});
			ii = permute(ii,[2 3 1 5 4]);
			ll = cns('get',{m.layer_map.label,'val'});
			ll = permute(ll,[2 3 1 5 4]);
			oo = cns('get',{m.layer_map.output,'val'});
			oo = permute(oo,[2 3 1 5 4]);
	[ll(:)';oo(:)']
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
keyboard

end

return


return
