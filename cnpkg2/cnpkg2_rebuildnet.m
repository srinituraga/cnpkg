function m = cnpkg_rebuildnet(m);

old_m = m;
m = rmfield(m,'layers');
m = rmfield(m,'layer_map');
m = rmfield(m,'step_map');

% build the gpu cns model structure
m = cnpkg2_buildmodel(m);

% copy in the old weights and biases
for l = 1:length(old_m.layer_map.weight),
	for s = 1:length(old_m.layer_map.weight{l}),
		for w = 1:length(old_m.layer_map.weight{l}{s}),
			m.layers{m.layer_map.weight{l}{s}(w)}.val(:) = ...
					old_m.layers{old_m.layer_map.weight{l}{s}(w)}.val(:);
			m.layers{m.layer_map.weight{l}{s}(w)}.eta = ...
					old_m.layers{old_m.layer_map.weight{l}{s}(w)}.eta;
		end
		m.layers{m.layer_map.bias{l}{s}}.val(:) = ...
				old_m.layers{old_m.layer_map.bias{l}{s}}.val(:);
		m.layers{m.layer_map.bias{l}{s}}.eta = ...
				old_m.layers{old_m.layer_map.bias{l}{s}}.eta;
	end
end

% mapdim
m = cnpkg2_mapdim_layers_bkwd(m,m.params.output_size,m.params.minibatch_size*2);
