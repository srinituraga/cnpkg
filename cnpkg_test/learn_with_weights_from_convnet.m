function m = learn_with_weights_from_convnet(m,n)

n.params.output_size = [1 1 1];
n.params.minibatch_size = 50;
m = cnpkg_buildmodel_from_convnet(n);

% number of iterations per block
nIter = 1e4;
nLoop1 = 1;
nLoop2 = 100;

for i = 1:length(m.layer_map.weight),
	for nHidOut = 1:size(n.layer{i}.W,6),
	for nHidIn = 1:size(n.layer{i}.W,4),
		m.layers{m.layer_map.weight(i)}.val(nHidIn,:,:,:,nHidOut) = n.layer{i}.W(:,:,:,nHidIn,1,nHidOut);
		m.layers{m.layer_map.weight(i)}.eta = mean(n.layer{i}.etaW(:));
	end
	end
	m.layers{m.layer_map.bias(i)}.val = n.layer{i}.B(:);
	m.layers{m.layer_map.bias(i)}.eta = mean(n.layer{i}.etaB(:));
end

%% load data and train a network
data = load(n.data_info.training_files{1});

loss = [];
for loop1 = 1:nLoop1,
	% initialize the indices
	fprintf('Picking images & indices...')
	tic,
	[m,segIdx,imValidSz] = pick_images(m,data,n,20);
	m = pick_indices(m,segIdx,imValidSz,nIter);
	fprintf(' done.\n')
	toc

	% initialize gpu
	fprintf('Initializing on gpu...')
	tic
	cns('init',m,'gpu','mean')
	fprintf(' done.\n')
	toc

% 	% run!
% 	fprintf('Running on gpu...')
% 	tic
% 	loss = cns('run',nIter,{m.layer_map.output,'loss'});
% 	loss = loss(:,:);
% 	fprintf(' done.\n')
% 	toc
% 	figure(2),plot(sqrt(smooth(mean(loss,2),1e3))), drawnow

	for loop2 = 1:nLoop2,
		fprintf('Picking indices...')
		tic,
		m = pick_indices(m,segIdx,imValidSz,nIter);
		m.layers{m.layer_map.minibatch_index}.val = repmat(m.layers{m.layer_map.minibatch_index}.val(:,1,:),[1 nIter 1]);
		cns('set',{0,'iter_no',0},{m.layer_map.minibatch_index,'val',m.layers{m.layer_map.minibatch_index}.val});
		fprintf(' done.\n')
		toc

		fprintf('Running on gpu...')
		tic
		[loss_block] = cns('run',nIter,{m.layer_map.output,'loss'});
		loss(end+1:end+nIter,:) = loss_block(:,:);
		fprintf(' done.\n')
		toc
		figure(2),plot(sqrt(smooth(mean(loss,2),1e3))), drawnow
	end
	m = cns('update',m);
end

keyboard



%% Helper functions!
% pick indices
function m = pick_indices(m,segIdx,imValidSz,nIter),
	nBatch = m.layers{m.layer_map.minibatch_index}.size{3};
	index = zeros(5,nIter,nBatch);

	index(4,:,:) = randi(length(m.layers{m.layer_map.input}.inputblock),[1 nIter,nBatch]);

	% loop over images
	for k = 1:length(imValidSz),
		[junk,idxIter,idxBatch] = ind2sub([1 nIter nBatch],find(index(4,:,:)==k));
		idx1 = randi(imValidSz{k}(1),[length(idxIter) 1]);
		idx2 = randi(imValidSz{k}(2),[length(idxIter) 1]);
		idx3 = randi(imValidSz{k}(3),[length(idxIter) 1]);
		idx5 = randi(length(segIdx{k}),[length(idxIter) 1]);
		index(sub2ind(size(index),repmat(1,[length(idxIter) 1]),idxIter,idxBatch)) = idx1;
		index(sub2ind(size(index),repmat(2,[length(idxIter) 1]),idxIter,idxBatch)) = idx2;
		index(sub2ind(size(index),repmat(3,[length(idxIter) 1]),idxIter,idxBatch)) = idx3;
		index(sub2ind(size(index),repmat(5,[length(idxIter) 1]),idxIter,idxBatch)) = idx5;
	end

	m.layers{m.layer_map.minibatch_index}.val = index-1; % convert to zero based indexing
	m.layers{m.layer_map.minibatch_index}.size{2} = nIter; % convert to zero based indexing
return

% pick a random subset of the images
function [m,segIdx,imValidSz] = pick_images(m,data,n,nIm)
	outputSz = cell2mat(m.layers{m.layer_map.output}.size(2:4));
	borderSz = 2*m.layers{m.layer_map.output}.offset;
	totalBorderSz = borderSz + outputSz - 1;
	segOffset = 0;
	imKeep = randsample(length(data.im),nIm);

	for iIm = 1:length(imKeep),
		m.layers{m.layer_map.input}.inputblock{iIm} = permute(data.im{imKeep(iIm)},[4 1 2 3]);
		imValidSz{iIm} = size(data.im{imKeep(iIm)});
		if length(imValidSz{iIm})<3,imValidSz{iIm}(length(imValidSz{iIm})+1:3)=1;end
		imValidSz{iIm} = imValidSz{iIm}(1:3) - totalBorderSz;
		nSeg = length(data.seg{imKeep(iIm)});
		segIdx{iIm} = segOffset + (1:nSeg);
		for iseg = 1:nSeg,
			m.layers{m.layer_map.output}.labelblock{segOffset + iseg} = ...
					permute(single(MakeConnLabel(data.seg{imKeep(iIm)}{iseg},n.params.nhood)),[4 1 2 3]);
			m.layers{m.layer_map.output}.maskblock{segOffset + iseg} = ...
					repmat( ...
					permute(single(data.mask{imKeep(iIm)}{iseg}),[4 1 2 3]), ...
					[size(m.layers{m.layer_map.output}.labelblock,1) 1 1 1]);
			m.layers{m.layer_map.output}.maskblock{segOffset + iseg}(:) = 1;
		end
		segOffset = segOffset + nSeg;
	end
return
return
