function m = cnpkg_buildmodel(m)

m.package = 'cnpkg';
m.independent = true;

%% Assign layer numbers to weights, biases, activities, etc
l = 0;
% fwd bkwd gradient passes
l = l(end) + 1;
m.layer_map.minibatch_index = l;
l = l(end) + 1;
m.layer_map.input = l;
l = l(end) + (1:(m.params.num_layers-1));
m.layer_map.hidden = l;
l = l(end) + 1;
m.layer_map.output = l;
l = l(end) + 1;
m.layer_map.label = l;
l = l(end) + (1:(m.params.num_layers));
m.layer_map.weight = l;
l = l(end) + (1:(m.params.num_layers));
m.layer_map.bias = l;
m.layer_map.hidden_prev_layer = [m.layer_map.input m.layer_map.hidden(1:end-1)];
m.layer_map.hidden_next_layer = [m.layer_map.hidden(2:end) m.layer_map.output];


%% Fill in the cnpkg layer parameters
% minibatch index layer
l = m.layer_map.minibatch_index;
m.layers{l}.name = 'xi';
m.layers{l}.type = 'index';
m.layers{l}.stepNo  = [];
m.layers{l}.size = {5, 1, m.params.minibatch_size*2};
% input layer
l = m.layer_map.input;
m.layers{l}.z       = l; % For display only.
m.layers{l}.name    = 'input_layer';
m.layers{l}.type    = 'input';
m.layers{l}.stepNo  = [];
m.layers{l}.size{1} = m.params.input_units;
m.layers{l}.size{5} = m.params.minibatch_size*2;
m.layers{l}.zin = m.layer_map.minibatch_index;
m.layers{l}.inputblock{1} = [];
% hidden layers
for i = 1:length(m.layer_map.hidden),
	l = m.layer_map.hidden(i);
    m.layers{l}.z       = l; % For display only.
    m.layers{l}.name    = sprintf('hidden_layer_%u', i);
    m.layers{l}.type    = 'hidden';
    m.layers{l}.stepNo  = [];
    m.layers{l}.zp      = m.layer_map.hidden_prev_layer(i);
    m.layers{l}.zw      = m.layer_map.weight(i);
    m.layers{l}.zb      = m.layer_map.bias(i);
    m.layers{l}.znw     = m.layer_map.weight(i+1);
    m.layers{l}.zn      = m.layer_map.hidden_next_layer(i);
    m.layers{l}.size{1} = m.params.layer{i}.nHid;
	m.layers{l}.size{5} = m.params.minibatch_size*2;

	% corresponding weights & biases
	l = m.layer_map.weight(i);
    m.layers{l}.z       = l; % For display only.
    m.layers{l}.name    = sprintf('weight_layer_%u', i);
    m.layers{l}.type    = 'weight';
    m.layers{l}.stepNo  = [];
    m.layers{l}.zp      = m.layer_map.hidden_prev_layer(i);
    m.layers{l}.zn      = m.layer_map.hidden(i);
    m.layers{l}.eta     = mean(m.params.layer{i}.etaW(:));
    m.layers{l}.size{1} = m.layers{m.layer_map.hidden_prev_layer(i)}.size{1};
    m.layers{l}.size{2} = m.params.layer{i}.patchSz(1);
    m.layers{l}.size{3} = m.params.layer{i}.patchSz(2);
    m.layers{l}.size{4} = m.params.layer{i}.patchSz(3);
    m.layers{l}.size{5} = m.params.layer{i}.nHid;
	m.layers{l}.val = single(m.params.layer{i}.initW*randn(cell2mat(m.layers{l}.size))/sqrt(prod(cell2mat(m.layers{l}.size(1:4)))));

	l = m.layer_map.bias(i);
    m.layers{l}.z       = l; % For display only.
    m.layers{l}.name    = sprintf('bias_layer_%u', i);
    m.layers{l}.type    = 'bias';
    m.layers{l}.stepNo  = [];
    m.layers{l}.zn      = m.layer_map.hidden(i);
    m.layers{l}.eta     = mean(m.params.layer{i}.etaB(:));
    m.layers{l}.size{1} = m.params.layer{i}.nHid;
    m.layers{l}.size{2} = 1;
	m.layers{l}.val = single(m.params.layer{i}.initB*randn(cell2mat(m.layers{l}.size)));
end
% output layer
l = m.layer_map.output;
m.layers{l}.z       = l; % For display only.
m.layers{l}.name    = sprintf('output');
m.layers{l}.type    = 'output';
m.layers{l}.stepNo  = [];
m.layers{l}.zp      = m.layer_map.hidden(end);
m.layers{l}.zw      = m.layer_map.weight(end);
m.layers{l}.zb      = m.layer_map.bias(end);
m.layers{l}.size{1} = m.params.output_units;
m.layers{l}.size{5} = m.params.minibatch_size*2;
m.layers{l}.zl = m.layer_map.label;
% label layer
l = m.layer_map.label;
m.layers{l}.z       = l; % For display only.
m.layers{l}.name    = sprintf('label');
m.layers{l}.type    = 'label';
m.layers{l}.stepNo  = [];
m.layers{l}.size{1} = m.params.output_units;
m.layers{l}.size{5} = m.params.minibatch_size*2;
m.layers{l}.zin = m.layer_map.minibatch_index;
m.layers{l}.labelblock{1} = [];
m.layers{l}.maskblock{1} = [];
% corresponding weights & biases
l = m.layer_map.weight(end);
m.layers{l}.z       = l; % For display only.
m.layers{l}.name    = sprintf('weight_layer_%u', length(m.layer_map.weight));
m.layers{l}.type    = 'weight';
m.layers{l}.stepNo  = [];
m.layers{l}.zp      = m.layer_map.hidden(end);
m.layers{l}.zn      = m.layer_map.output;
m.layers{l}.eta     = mean(m.params.layer{end}.etaW(:));
m.layers{l}.size{1} = m.layers{m.layer_map.hidden(end)}.size{1};
m.layers{l}.size{2} = m.params.layer{end}.patchSz(1);
m.layers{l}.size{3} = m.params.layer{end}.patchSz(2);
m.layers{l}.size{4} = m.params.layer{end}.patchSz(3);
m.layers{l}.size{5} = m.params.output_units;
m.layers{l}.val = single(m.params.layer{end}.initW*randn(cell2mat(m.layers{l}.size))/sqrt(prod(cell2mat(m.layers{l}.size(1:4)))));
% biases
l = m.layer_map.bias(end);
m.layers{l}.z       = l; % For display only.
m.layers{l}.name    = sprintf('bias_layer_%u', length(m.layer_map.bias));
m.layers{l}.type    = 'bias';
m.layers{l}.stepNo  = [];
m.layers{l}.zn      = m.layer_map.output;
m.layers{l}.eta     = mean(m.params.layer{end}.etaB(:));
m.layers{l}.size{1} = m.params.output_units;
m.layers{l}.size{2} = 1;
m.layers{l}.val = single(m.params.layer{end}.initB*randn(cell2mat(m.layers{l}.size)));


%% Setup step nos

step = 0;
% fwd pass
step = step+1;
m.step_map.fwd(1) = step;
l = m.layer_map.input;
m.layers{l}.stepNo = [m.layers{l}.stepNo step];
for i = 1:length(m.layer_map.hidden),
	step = step+1;
	l = m.layer_map.hidden(i);
	m.layers{l}.stepNo = [m.layers{l}.stepNo step];
end
step = step+1;
l = m.layer_map.output;
m.layers{l}.stepNo = [m.layers{l}.stepNo step];
m.step_map.fwd(2) = step;

% bkwd pass
step = step+1;
m.step_map.bkwd(1) = step;
l = m.layer_map.label;
m.layers{l}.stepNo = [m.layers{l}.stepNo step];
step = step+1;
l = m.layer_map.output;
m.layers{l}.stepNo = [m.layers{l}.stepNo step];
for i = length(m.layer_map.hidden):-1:1,
	step = step+1;
	l = m.layer_map.hidden(i);
	m.layers{l}.stepNo = [m.layers{l}.stepNo step];
end
m.step_map.bkwd(2) = step;

% gradient pass
step = step+1;
m.step_map.gradient = step;
for i = 1:length(m.layer_map.weight),
	l = m.layer_map.weight(i);
	m.layers{l}.stepNo = [m.layers{l}.stepNo step];
	l = m.layer_map.bias(i);
	m.layers{l}.stepNo = [m.layers{l}.stepNo step];
end


return;
