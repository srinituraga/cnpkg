function m = buildModel(m)

fields = {'layers','layer_map','step_map'};
for k = 1:length(fields),
    if isfield(m,fields{k}),
        m = rmfield(m,fields{k});
    end
end

%% CNS package to use
m.package = 'cnpkg4';
m.independent = true;

m.inputblock = {[]};
m.labelblock = {[]};
m.maskblock = {[]};
m.binarythreshold = 0.5;

dims = ['y' 'x' 'd'];
%% Assign layer numbers to weights, biases, activities, etc
l = 0;
% fwd bkwd gradient passes
l = l(end) + 1;
layer_map.minibatch_index = l;
l = l(end) + 1;
layer_map.input = l;
for h = 1:(m.params.num_layers-1),
	l = l(end) + (1:length(m.params.layer{h}.nHid));
	layer_map.computed{h} = l;
	l = l(end) + (1:length(m.params.layer{h}.nHid));
	layer_map.sens{h} = l;
end
l = l(end) + 1;
layer_map.computed{end+1} = l;
layer_map.output = l;
l = l(end) + 1;
layer_map.error = l; layer_map.sens{end+1} = l;
l = l(end) + 1;
layer_map.label = l;
layer_map.computed_prev{1} = layer_map.input;
for h = 2:length(layer_map.computed),
	layer_map.computed_prev{h} = layer_map.computed{h-1};
end
layer_map.computed_next = {};
layer_map.sens_next = {};
for h = 1:length(layer_map.computed)-1,
	layer_map.computed_next{h} = layer_map.computed{h+1};
	layer_map.sens_next{h} = layer_map.sens{h+1};
end

% weights
for h = 1:length(layer_map.computed),
	for s = 1:length(layer_map.computed{h}),
		l = l(end) + (1:length(layer_map.computed_prev{h}));
		layer_map.weight{h}{s} = l;
		l = l(end) + 1;
		layer_map.bias{h}{s} = l;
	end
end
for h = 1:length(layer_map.weight)-1,
	for s = 1:length(layer_map.weight{h}),
		layer_map.weight_out{h}{s} = [];
		for sn = 1:length(layer_map.weight{h+1}),
			layer_map.weight_out{h}{s} = [layer_map.weight_out{h}{s} layer_map.weight{h+1}{sn}(s)];
		end
	end
end


%% Fill in the cnpkg layer parameters
% minibatch index layer
l = layer_map.minibatch_index;
m.layers{l}.name = 'xi';
m.layers{l}.type = 'index';
m.layers{l}.size{1} = 5;
m.layers{l}.size{2} = 1;
m.layers{l}.val = 0;
% input layer
l = layer_map.input;
m.layers{l}.z       = l; % For display only.
m.layers{l}.name    = 'inputLayer';
m.layers{l}.type    = 'input';
m.layers{l}.size{1} = m.params.input_units;
for kk=1:3,
    m.layers{l}.([dims(kk) '_space']) = 1;
end
m.layers{l}.zin		= layer_map.minibatch_index;
m.layers{l}.zn      = layer_map.computed{1};
% hidden layers
for i = 1:length(layer_map.computed)-1,
	for s = 1:length(layer_map.computed{i}),
		l = layer_map.computed{i}(s);
		m.layers{l}.z       = l; % For display only.
		m.layers{l}.name    = ['hiddenLayerNo ' num2str(i) ', scaleNo ' num2str(s)];
		m.layers{l}.type    = 'computed';
		m.layers{l}.zp      = layer_map.computed_prev{i};
		m.layers{l}.zw      = layer_map.weight{i}{s};
		m.layers{l}.zb      = layer_map.bias{i}{s};
		m.layers{l}.zn      = layer_map.computed_next{i};
		m.layers{l}.size{1} = m.params.layer{i}.nHid{s};
        for kk=1:3,
            m.layers{l}.([dims(kk) '_space']) = m.params.layer{i}.scales{s}(kk);
        end

		l = layer_map.sens{i}(s);
		m.layers{l}.z       = l; % For display only.
		m.layers{l}.name    = ['hiddenLayerNo ' num2str(i) ', scaleNo ' num2str(s)];
		m.layers{l}.type    = 'sens';
		m.layers{l}.zme     = layer_map.computed{i}(s);
		m.layers{l}.znw     = layer_map.weight_out{i}{s};
		m.layers{l}.zn      = layer_map.sens_next{i};
		m.layers{l}.size{1} = m.params.layer{i}.nHid{s};
        for kk=1:3,
            m.layers{l}.([dims(kk) '_space']) = m.params.layer{i}.scales{s}(kk);
        end

		% corresponding weights & biases
		for sp = 1:length(layer_map.computed_prev{i}),
			l = layer_map.weight{i}{s}(sp);
			m.layers{l}.z       = l; % For display only.
			m.layers{l}.name    = ['weightLayer ' num2str(i) ', scaleNo' num2str(s) 'prevScaleNo' num2str(sp)];
			m.layers{l}.type    = 'weight';
            m.layers{l}.kernel  = {'','constrain'};
            m.layers{l}.convType= 'valid';
			m.layers{l}.zp      = layer_map.computed_prev{i}(sp);
			m.layers{l}.zs      = layer_map.sens{i}(s);
			m.layers{l}.eta     = m.params.layer{i}.etaW{s};
			if ~isfield(m.params.layer{i},'wPxSize') || (s>length(m.params.layer{i}.wPxSize)),
                m.params.layers{i}.wPxSize{s} = [1 1 1];
			end
            for kk=1:3,
                m.layers{l}.([dims(kk) '_blk']) = m.params.layer{i}.wPxSize{s}(kk);
            end
            for kk=1:3,
                m.layers{l}.([dims(kk) '_space']) = 1;
            end
			m.layers{l}.size{1} = m.layers{layer_map.computed_prev{i}(sp)}.size{1};
			m.layers{l}.size{2} = m.params.layer{i}.patchSz{s}(1)*m.layers{l}.y_blk;
			m.layers{l}.size{3} = m.params.layer{i}.patchSz{s}(2)*m.layers{l}.x_blk;
			m.layers{l}.size{4} = m.params.layer{i}.patchSz{s}(3)*m.layers{l}.d_blk;
			m.layers{l}.size{5} = m.layers{layer_map.computed{i}(s)}.size{1};
			m.layers{l}.val		= single(m.params.layer{i}.initW{s}*randn(cell2mat(m.layers{l}.size))/sqrt(prod(cell2mat(m.layers{l}.size(1:4)))));
			m.layers{l}.val		= upSmpl(dnSmplMean(m.layers{l}.val,[1 m.params.layer{i}.wPxSize{s} 1]),[1 m.params.layer{i}.wPxSize{s} 1]);
            m.layers{l}.dval = zeros(size(m.layers{l}.val));
		end

		l = layer_map.bias{i}{s};
		m.layers{l}.z       = l; % For display only.
		m.layers{l}.name    = ['biasLayer', num2str(i) ', scaleNo' num2str(s)];
		m.layers{l}.type    = 'bias';
		m.layers{l}.zs      = layer_map.sens{i}(s);
		m.layers{l}.eta     = m.params.layer{i}.etaB{s};
		m.layers{l}.size{1} = m.layers{layer_map.computed{i}(s)}.size{1};
		m.layers{l}.size{2} = 1;
		m.layers{l}.val		= single(m.params.layer{i}.initB{s}*randn(cell2mat(m.layers{l}.size)));
        m.layers{l}.dval = zeros(size(m.layers{l}.val));
	end
end
% output layer
l = layer_map.computed{end};
m.layers{l}.z       = l; % For display only.
m.layers{l}.name    = sprintf('output');
m.layers{l}.type    = 'computed';
m.layers{l}.zp      = layer_map.computed{end-1};
m.layers{l}.zw      = layer_map.weight{end}{1};
m.layers{l}.zb      = layer_map.bias{end}{1};
m.layers{l}.zn      = [];
m.layers{l}.size{1} = m.params.output_units;
for kk=1:3,
    m.layers{l}.([dims(kk) '_space']) = 1;
end
% error layer
l = layer_map.error;
m.layers{l}.z       = l; % For display only.
m.layers{l}.name    = sprintf('error');
m.layers{l}.type    = 'error';
m.layers{l}.zme     = layer_map.computed{end};
m.layers{l}.zn      = [];
m.layers{l}.znw     = [];
m.layers{l}.size{1} = m.params.output_units;
m.layers{l}.zl		= layer_map.label;
for kk=1:3,
    m.layers{l}.([dims(kk) '_space']) = 1;
end
% label layer
l = layer_map.label;
m.layers{l}.z       = l; % For display only.
m.layers{l}.name    = sprintf('label');
m.layers{l}.type    = 'label';
m.layers{l}.zin		= layer_map.minibatch_index;
m.layers{l}.size{1} = m.params.output_units;
for kk=1:3,
    m.layers{l}.([dims(kk) '_space']) = 1;
end
% corresponding weights & biases
for sp = 1:length(layer_map.computed{end-1}),
	l = layer_map.weight{end}{1}(sp);
	m.layers{l}.z       = l; % For display only.
	m.layers{l}.name    = ['weightLayer', num2str(length(layer_map.weight))];
	m.layers{l}.type    = 'weight';
    m.layers{l}.kernel  = {'','constrain'};
    m.layers{l}.convType= 'valid';
	m.layers{l}.zp      = layer_map.computed{end-1}(sp);
	m.layers{l}.zs      = layer_map.error;
	m.layers{l}.eta     = m.params.layer{end}.etaW{1};
    if ~isfield(m.params.layer{end},'wPxSize'),
        m.params.layer{end}.wPxSize{1} = [1 1 1];
    end
    for kk=1:3,
        m.layers{l}.([dims(kk) '_blk']) = m.params.layer{end}.wPxSize{1}(kk);
    end
    for kk=1:3,
        m.layers{l}.([dims(kk) '_space']) = 1;
    end
	m.layers{l}.size{1} = m.layers{layer_map.computed{end-1}(sp)}.size{1};
	m.layers{l}.size{2} = m.params.layer{end}.patchSz{1}(1)*m.layers{l}.y_blk;
	m.layers{l}.size{3} = m.params.layer{end}.patchSz{1}(2)*m.layers{l}.x_blk;
	m.layers{l}.size{4} = m.params.layer{end}.patchSz{1}(3)*m.layers{l}.d_blk;
	m.layers{l}.size{5} = m.params.output_units;
    m.layers{l}.val		= single(m.params.layer{end}.initW{1}*randn(cell2mat(m.layers{l}.size))/sqrt(prod(cell2mat(m.layers{l}.size(1:4)))));
    m.layers{l}.val		= upSmpl(dnSmplMean(m.layers{l}.val,[1 m.params.layer{end}.wPxSize{1} 1]),[1 m.params.layer{end}.wPxSize{1} 1]);
    m.layers{l}.dval   = zeros(size(m.layers{l}.val));
end
% biases
l = layer_map.bias{end}{1};
m.layers{l}.z       = l; % For display only.
m.layers{l}.name    = ['biasLayer' num2str(length(layer_map.bias))];
m.layers{l}.type    = 'bias';
m.layers{l}.zs      = layer_map.error;
m.layers{l}.eta     = m.params.layer{end}.etaB{1};
m.layers{l}.size{1} = m.params.output_units;
m.layers{l}.size{2} = 1;
m.layers{l}.val		= single(m.params.layer{end}.initB{1}*randn(cell2mat(m.layers{l}.size)));
m.layers{l}.dval   = zeros(size(m.layers{l}.val));

m.layer_map = layer_map;
