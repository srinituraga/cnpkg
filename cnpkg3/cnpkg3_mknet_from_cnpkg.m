function m = cnpkg_mknet(m);

m.old_m = m;
m = rmfield(m,'layers');
m = rmfield(m,'layer_map');
m = rmfield(m,'step_map');
m.params = rmfield(m.params,'layer');

% structure of each layer
for l = 1:m.params.num_layers-1,
	m.params.layer{l}.nHid{1} = m.old_m.params.layer{l}.nHid;
	m.params.layer{l}.patchSz{1} = m.old_m.params.layer{l}.patchSz;
	m.params.layer{l}.scales{1} = [1 1 1];
	m.params.layer{l}.wPxSize{1} = [1 1 1];
	m.params.layer{l}.etaW{1} = m.old_m.params.layer{l}.etaW;
	m.params.layer{l}.etaB{1} = m.old_m.params.layer{l}.etaB;
	m.params.layer{l}.initW{1} = m.old_m.params.layer{l}.initW;
	m.params.layer{l}.initB{1} = m.old_m.params.layer{l}.initB;
end
for l = m.params.num_layers,
	m.params.layer{l}.patchSz{1} = m.old_m.params.layer{l}.patchSz;
	m.params.layer{l}.scales{1} = [1 1 1];
	m.params.layer{l}.wPxSize{1} = [1 1 1];
	m.params.layer{l}.etaW{1} = m.old_m.params.layer{l}.etaW;
	m.params.layer{l}.etaB{1} = m.old_m.params.layer{l}.etaB;
	m.params.layer{l}.initW{1} = m.old_m.params.layer{l}.initW;
	m.params.layer{l}.initB{1} = m.old_m.params.layer{l}.initB;
end

% build the gpu cns model structure
m = cnpkg3_buildmodel(m);
m = cnpkg3_mapdim_layers_bkwd(m,m.params.output_size,m.params.minibatch_size*2);

% copy in the old weights
for l = 1:length(m.layer_map.weight),
	m.layers{m.layer_map.weight{l}{1}(1)}.val = ...
			m.old_m.layers{m.old_m.layer_map.weight(l)}.val;
	m.layers{m.layer_map.bias{l}{1}}.val = ...
			m.old_m.layers{m.old_m.layer_map.bias(l)}.val;
end
