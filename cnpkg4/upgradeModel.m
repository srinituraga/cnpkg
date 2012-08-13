function [m layer_map] = upgradeModel(oldm)

m.params = oldm.params;

% convert the params
for l=1:length(oldm.params.layer),
    if ~isfield(m.params.layer,'wPxSize'),
        for s=1:length(oldm.params.layer{l}.scales),
            m.params.layer{l}.wPxSize{s} = [1 1 1];
        end
    end
end
m = buildModel(m);

% copy in the old weights and biases
for l = 1:length(oldm.layer_map.weight),
	for s = 1:length(oldm.layer_map.weight{l}),
		for w = 1:length(oldm.layer_map.weight{l}{s}),
			m.layers{m.layer_map.weight{l}{s}(w)}.val(:) = ...
					oldm.layers{oldm.layer_map.weight{l}{s}(w)}.val(:);
			m.layers{m.layer_map.weight{l}{s}(w)}.eta = 1;
		end
		m.layers{m.layer_map.bias{l}{s}}.val(:) = ...
				oldm.layers{oldm.layer_map.bias{l}{s}}.val(:);
		m.layers{m.layer_map.bias{l}{s}}.eta = 1;
	end
end

fields = {'offset','leftBorder','rightBorder','totalBorder'};
for k=1:length(fields),
    if isfield(oldm,fields{k}),
        m.(fields{k}) = oldm.(fields{k});
    end
end

if ~isfield(m.layers{m.layer_map.error},'param'),
    m.layers{m.layer_map.error}.param = 0.3;
end
