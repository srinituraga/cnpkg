function m = cnpkg_mknet(n);

m.old_n = n;

m.params.minibatch_size = n.params.minibatch_size;
m.params.output_size = n.params.output_size;

% if minimax network, copy minimax params
if isfield(n.params,'graph_size'),
	m.params.graph_size = n.params.graph_size;
	m.params.constrained_minimax = n.params.constrained_minimax;
	m.params.nhood = n.params.nhood;
end

% max # of total training iterations (not epochs)
m.params.maxIter = n.params.maxIter;
m.params.nIterPerEpoch = 1e4;
m.params.nEpochPerSave = 1e1;


%%% NETWORK ARCHITECTURE %%%
m.params.input_units = n.params.input_units;
m.params.output_units = n.params.output_units;
m.params.num_layers = n.params.layers;

% structure of each layer
m.globaleta = 1e-0;
for l = 1:m.params.num_layers,
	m.params.layer{l}.nHid{1} = n.layer{l}.nHid;
	m.params.layer{l}.patchSz{1} = n.layer{l}.patchSz;
	m.params.layer{l}.scales{1} = [1 1 1];
	m.params.layer{l}.etaW{1} = mean(n.layer{l}.etaW(:));
	m.params.layer{l}.etaB{1} = mean(n.layer{l}.etaB(:));
	m.params.layer{l}.initW{1} = 0e-1;
	m.params.layer{l}.initB{1} = 0e-0;
end


% build the gpu cns model structure
m = cnpkg3_buildmodel(m);
m = cnpkg3_mapdim_layers_bkwd(m,m.params.output_size,m.params.minibatch_size*2);

% copy in the old weights
for l = 1:m.params.num_layers,
	for nHidOut = 1:size(n.layer{l}.W,6),
	for nHidIn = 1:size(n.layer{l}.W,4),
		m.layers{m.layer_map.weight{l}{1}(1)}.val(nHidIn,:,:,:,nHidOut) = n.layer{l}.W(:,:,:,nHidIn,1,nHidOut);
	end
	end
	m.layers{m.layer_map.bias{l}{1}(1)}.val = n.layer{l}.B(:);
end
