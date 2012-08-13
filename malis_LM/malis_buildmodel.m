function m = cnpkg_buildmodel(m)

m.package = 'cnpkg2';
m.independent = true;

%% Assign layer numbers to weights, biases, activities, etc
l = 0;
% fwd bkwd gradient passes
l = l(end) + 1;
m.layer_map.minibatch_index = l;
l = l(end) + 1;
m.layer_map.input = l;
for h = 1:(m.params.num_layers-1),
	l = l(end) + (1:length(m.params.layer{h}.nHid));
	m.layer_map.hidden{h} = l;
end
l = l(end) + 1;
m.layer_map.output = l;
l = l(end) + 1;
m.layer_map.label = l;
m.layer_map.hidden_prev{1} = m.layer_map.input;
for h = 2:length(m.layer_map.hidden),
	m.layer_map.hidden_prev{h} = m.layer_map.hidden{h-1};
end
m.layer_map.hidden_next = {};
for h = 1:length(m.layer_map.hidden)-1,
	m.layer_map.hidden_next{h} = m.layer_map.hidden{h+1};
end
m.layer_map.hidden_next{end+1} = m.layer_map.output;

% bigpos fwd pass
l = l(end) + 1;
m.layer_map.bigpos_minibatch_index = l;
l = l(end) + 1;
m.layer_map.bigpos_input = l;
for h = 1:(m.params.num_layers-1),
	l = l(end) + (1:length(m.params.layer{h}.nHid));
	m.layer_map.bigpos_hidden{h} = l;
end
l = l(end) + 1;
m.layer_map.bigpos_output = l;
l = l(end) + 1;
m.layer_map.bigpos_label = l;
m.layer_map.bigpos_hidden_prev{1} = m.layer_map.bigpos_input;
for h = 2:length(m.layer_map.bigpos_hidden),
	m.layer_map.bigpos_hidden_prev{h} = m.layer_map.bigpos_hidden{h-1};
end
m.layer_map.bigpos_hidden_next = {};
for h = 1:length(m.layer_map.bigpos_hidden)-1,
	m.layer_map.bigpos_hidden_next{h} = m.layer_map.bigpos_hidden{h+1};
end
m.layer_map.bigpos_hidden_next{end+1} = m.layer_map.bigpos_output;

% bigneg fwd pass
l = l(end) + 1;
m.layer_map.bigneg_minibatch_index = l;
l = l(end) + 1;
m.layer_map.bigneg_input = l;
for h = 1:(m.params.num_layers-1),
	l = l(end) + (1:length(m.params.layer{h}.nHid));
	m.layer_map.bigneg_hidden{h} = l;
end
l = l(end) + 1;
m.layer_map.bigneg_output = l;
l = l(end) + 1;
m.layer_map.bigneg_label = l;
m.layer_map.bigneg_hidden_prev{1} = m.layer_map.bigneg_input;
for h = 2:length(m.layer_map.bigneg_hidden),
	m.layer_map.bigneg_hidden_prev{h} = m.layer_map.bigneg_hidden{h-1};
end
m.layer_map.bigneg_hidden_next = {};
for h = 1:length(m.layer_map.bigneg_hidden)-1,
	m.layer_map.bigneg_hidden_next{h} = m.layer_map.bigneg_hidden{h+1};
end
m.layer_map.bigneg_hidden_next{end+1} = m.layer_map.bigneg_output;

% weights
for h = 1:length(m.layer_map.hidden),
	for s = 1:length(m.layer_map.hidden{h}),
		l = l(end) + (1:length(m.layer_map.hidden_prev{h}));
		m.layer_map.weight{h}{s} = l;
		l = l(end) + 1;
		m.layer_map.bias{h}{s} = l;
	end
end
l = l(end) + (1:length(m.layer_map.hidden{end}));
m.layer_map.weight{end+1}{1} = l;
l = l(end) + 1;
m.layer_map.bias{end+1}{1} = l;
for h = 1:length(m.layer_map.weight)-1,
	for s = 1:length(m.layer_map.weight{h}),
		m.layer_map.weight_out{h}{s} = [];
		for sn = 1:length(m.layer_map.weight{h+1}),
			m.layer_map.weight_out{h}{s} = [m.layer_map.weight_out{h}{s} m.layer_map.weight{h+1}{sn}(s)];
		end
	end
end


%% Fill in the cnpkg layer parameters
% minibatch index layer
l = m.layer_map.minibatch_index;
m.layers{l}.name = 'xi';
m.layers{l}.type = 'index';
m.layers{l}.stepNo  = [];
m.layers{l}.size{1} = 5;
m.layers{l}.size{2} = 1;
% input layer
l = m.layer_map.input;
m.layers{l}.z       = l; % For display only.
m.layers{l}.name    = 'inputLayer';
m.layers{l}.type    = 'input';
m.layers{l}.stepNo  = [];
m.layers{l}.size{1} = m.params.input_units;
m.layers{l}.zin		= m.layer_map.minibatch_index;
m.layers{l}.inputblock = {};
% hidden layers
for i = 1:length(m.layer_map.hidden),
	for s = 1:length(m.layer_map.hidden{i}),
		l = m.layer_map.hidden{i}(s);
		m.layers{l}.z       = l; % For display only.
		m.layers{l}.name    = ['hiddenLayerNo ' num2str(i) ', scaleNo ' num2str(s)];
		m.layers{l}.type    = 'hidden';
		m.layers{l}.stepNo  = [];
		m.layers{l}.zp      = m.layer_map.hidden_prev{i};
		m.layers{l}.zw      = m.layer_map.weight{i}{s};
		m.layers{l}.zb      = m.layer_map.bias{i}{s};
		m.layers{l}.znw     = m.layer_map.weight_out{i}{s};
		m.layers{l}.zn      = m.layer_map.hidden_next{i};
		m.layers{l}.size{1} = m.params.layer{i}.nHid{s};

		% corresponding weights & biases
		for sp = 1:length(m.layer_map.hidden_prev{i}),
			l = m.layer_map.weight{i}{s}(sp);
			m.layers{l}.z       = l; % For display only.
			m.layers{l}.name    = ['weightLayer ' num2str(i) ', scaleNo' num2str(s) 'prevScaleNo' num2str(sp)];
			m.layers{l}.type    = 'weight';
			m.layers{l}.stepNo  = [];
			m.layers{l}.zp      = m.layer_map.hidden_prev{i}(sp);
			m.layers{l}.zn      = m.layer_map.hidden{i}(s);
			m.layers{l}.eta     = m.params.layer{i}.etaW{s};
			if isfield(m.params.layer{i},'wPxSize'),
				m.layers{l}.wPxSize = m.params.layer{i}.wPxSize{s};
			else,
				m.layers{l}.wPxSize = [1 1 1];
			end			
			m.layers{l}.size{1} = m.layers{m.layer_map.hidden_prev{i}(sp)}.size{1};
			m.layers{l}.size{2} = m.params.layer{i}.patchSz{s}(1)*m.layers{l}.wPxSize(1);
			m.layers{l}.size{3} = m.params.layer{i}.patchSz{s}(2)*m.layers{l}.wPxSize(2);
			m.layers{l}.size{4} = m.params.layer{i}.patchSz{s}(3)*m.layers{l}.wPxSize(3);
			m.layers{l}.size{5} = m.layers{m.layer_map.hidden{i}(s)}.size{1};
			m.layers{l}.val		= single(m.params.layer{i}.initW{s}*randn(cell2mat(m.layers{l}.size))/sqrt(prod(cell2mat(m.layers{l}.size(1:4)))));
		end

		l = m.layer_map.bias{i}{s};
		m.layers{l}.z       = l; % For display only.
		m.layers{l}.name    = ['biasLayer', num2str(i) ', scaleNo' num2str(s)];
		m.layers{l}.type    = 'bias';
		m.layers{l}.stepNo  = [];
		m.layers{l}.zn      = m.layer_map.hidden{i}(s);
		m.layers{l}.eta     = m.params.layer{i}.etaB{s};
		m.layers{l}.size{1} = m.layers{m.layer_map.hidden{i}(s)}.size{1};
		m.layers{l}.size{2} = 1;
		m.layers{l}.val		= single(m.params.layer{i}.initB{s}*randn(cell2mat(m.layers{l}.size)));
	end
end
% output layer
l = m.layer_map.output;
m.layers{l}.z       = l; % For display only.
m.layers{l}.name    = sprintf('output');
m.layers{l}.type    = 'output';
m.layers{l}.stepNo  = [];
m.layers{l}.zp      = m.layer_map.hidden{end};
m.layers{l}.zw      = m.layer_map.weight{end}{1};
m.layers{l}.zb      = m.layer_map.bias{end}{1};
m.layers{l}.size{1} = m.params.output_units;
m.layers{l}.zl		= m.layer_map.label;
% label layer
l = m.layer_map.label;
m.layers{l}.z       = l; % For display only.
m.layers{l}.name    = sprintf('label');
m.layers{l}.type    = 'label';
m.layers{l}.stepNo  = [];
m.layers{l}.size{1} = m.params.output_units;
m.layers{l}.zin		= m.layer_map.minibatch_index;
m.layers{l}.labelblock = {};
m.layers{l}.maskblock = {};
% corresponding weights & biases
for sp = 1:length(m.layer_map.hidden{end}),
	l = m.layer_map.weight{end}{1}(sp);
	m.layers{l}.z       = l; % For display only.
	m.layers{l}.name    = ['weightLayer', num2str(length(m.layer_map.weight))];
	m.layers{l}.type    = 'weight';
	m.layers{l}.stepNo  = [];
	m.layers{l}.zp      = m.layer_map.hidden{end}(sp);
	m.layers{l}.zn      = m.layer_map.output;
	m.layers{l}.eta     = m.params.layer{end}.etaW{1};
	if isfield(m.params.layer{end},'wPxSize'),
		m.layers{l}.wPxSize = m.params.layer{end}.wPxSize{1};
	else,
		m.layers{l}.wPxSize = [1 1 1];
	end	
	m.layers{l}.size{1} = m.layers{m.layer_map.hidden{end}(sp)}.size{1};
	m.layers{l}.size{2} = m.params.layer{end}.patchSz{1}(1)*m.layers{l}.wPxSize(1);
	m.layers{l}.size{3} = m.params.layer{end}.patchSz{1}(2)*m.layers{l}.wPxSize(2);
	m.layers{l}.size{4} = m.params.layer{end}.patchSz{1}(3)*m.layers{l}.wPxSize(3);
	m.layers{l}.size{5} = m.params.output_units;
	m.layers{l}.val		= single(m.params.layer{end}.initW{1}*randn(cell2mat(m.layers{l}.size))/sqrt(prod(cell2mat(m.layers{l}.size(1:4)))));
end
% biases
l = m.layer_map.bias{end}{1};
m.layers{l}.z       = l; % For display only.
m.layers{l}.name    = ['biasLayer' num2str(length(m.layer_map.bias))];
m.layers{l}.type    = 'bias';
m.layers{l}.stepNo  = [];
m.layers{l}.zn      = m.layer_map.output;
m.layers{l}.eta     = m.params.layer{end}.etaB{1};
m.layers{l}.size{1} = m.params.output_units;
m.layers{l}.size{2} = 1;
m.layers{l}.val		= single(m.params.layer{end}.initB{1}*randn(cell2mat(m.layers{l}.size)));

% bigpos minibatch index layer
l = m.layer_map.bigpos_minibatch_index;
m.layers{l}.name = 'bigpos index';
m.layers{l}.type = 'index';
m.layers{l}.stepNo  = [];
m.layers{l}.size{1} = 5;
m.layers{l}.size{2} = 1;
% input layer
l = m.layer_map.bigpos_input;
m.layers{l}.z       = l; % For display only.
m.layers{l}.name    = 'bigpos inputLayer';
m.layers{l}.type    = 'input';
m.layers{l}.stepNo  = [];
m.layers{l}.size{1} = m.params.input_units;
m.layers{l}.zin		= m.layer_map.bigpos_minibatch_index;
m.layers{l}.inputblock = {};
% hidden layers
for i = 1:length(m.layer_map.bigpos_hidden),
	for s = 1:length(m.layer_map.bigpos_hidden{i}),
		l = m.layer_map.bigpos_hidden{i}(s);
		m.layers{l}.z       = l; % For display only.
		m.layers{l}.name    = ['bigpos hiddenLayerNo ' num2str(i) ', scaleNo ' num2str(s)];
		m.layers{l}.type    = 'hidden';
		m.layers{l}.stepNo  = [];
		m.layers{l}.zp      = m.layer_map.bigpos_hidden_prev{i};
		m.layers{l}.zw      = m.layer_map.weight{i}{s};
		m.layers{l}.zb      = m.layer_map.bias{i}{s};
		m.layers{l}.znw     = m.layer_map.weight_out{i}{s};
		m.layers{l}.zn      = m.layer_map.bigpos_hidden_next{i};
		m.layers{l}.size{1} = m.params.layer{i}.nHid{s};
	end
end
% output layer
l = m.layer_map.bigpos_output;
m.layers{l}.z       = l; % For display only.
m.layers{l}.name    = sprintf('bigpos output');
m.layers{l}.type    = 'output';
m.layers{l}.stepNo  = [];
m.layers{l}.zp      = m.layer_map.bigpos_hidden{end};
m.layers{l}.zw      = m.layer_map.weight{end}{1};
m.layers{l}.zb      = m.layer_map.bias{end}{1};
m.layers{l}.size{1} = m.params.output_units;
m.layers{l}.zl		= m.layer_map.label;
% label layer
l = m.layer_map.bigpos_label;
m.layers{l}.z       = l; % For display only.
m.layers{l}.name    = ['bigpos label'];
m.layers{l}.type    = 'label';
m.layers{l}.stepNo  = [];
m.layers{l}.size{1} = m.params.output_units;
m.layers{l}.zin		= m.layer_map.minibatch_index;
m.layers{l}.labelblock{1} = {};
m.layers{l}.maskblock{1} = {};

% bigneg minibatch index layer
l = m.layer_map.bigneg_minibatch_index;
m.layers{l}.name = 'bigneg index';
m.layers{l}.type = 'index';
m.layers{l}.stepNo  = [];
m.layers{l}.size{1} = 5;
m.layers{l}.size{2} = 1;
% input layer
l = m.layer_map.bigneg_input;
m.layers{l}.z       = l; % For display only.
m.layers{l}.name    = 'bigneg inputLayer';
m.layers{l}.type    = 'input';
m.layers{l}.stepNo  = [];
m.layers{l}.size{1} = m.params.input_units;
m.layers{l}.zin		= m.layer_map.bigneg_minibatch_index;
m.layers{l}.inputblock = {};
% hidden layers
for i = 1:length(m.layer_map.bigneg_hidden),
	for s = 1:length(m.layer_map.bigneg_hidden{i}),
		l = m.layer_map.bigneg_hidden{i}(s);
		m.layers{l}.z       = l; % For display only.
		m.layers{l}.name    = ['bigneg hiddenLayerNo ' num2str(i) ', scaleNo ' num2str(s)];
		m.layers{l}.type    = 'hidden';
		m.layers{l}.stepNo  = [];
		m.layers{l}.zp      = m.layer_map.bigneg_hidden_prev{i};
		m.layers{l}.zw      = m.layer_map.weight{i}{s};
		m.layers{l}.zb      = m.layer_map.bias{i}{s};
		m.layers{l}.znw     = m.layer_map.weight_out{i}{s};
		m.layers{l}.zn      = m.layer_map.bigneg_hidden_next{i};
		m.layers{l}.size{1} = m.params.layer{i}.nHid{s};
	end
end
% output layer
l = m.layer_map.bigneg_output;
m.layers{l}.z       = l; % For display only.
m.layers{l}.name    = sprintf('bigneg output');
m.layers{l}.type    = 'output';
m.layers{l}.stepNo  = [];
m.layers{l}.zp      = m.layer_map.bigneg_hidden{end};
m.layers{l}.zw      = m.layer_map.weight{end}{1};
m.layers{l}.zb      = m.layer_map.bias{end}{1};
m.layers{l}.size{1} = m.params.output_units;
m.layers{l}.zl		= m.layer_map.label;
% label layer
l = m.layer_map.bigneg_label;
m.layers{l}.z       = l; % For display only.
m.layers{l}.name    = ['bigneg label'];
m.layers{l}.type    = 'label';
m.layers{l}.stepNo  = [];
m.layers{l}.size{1} = m.params.output_units;
m.layers{l}.zin		= m.layer_map.minibatch_index;
m.layers{l}.labelblock{1} = {};
m.layers{l}.maskblock{1} = {};

%% Setup step nos

step = 0;
% bigpos fwd pass
step = step+1;
m.step_map.bigpos_fwd(1) = step;
l = m.layer_map.bigpos_input;
m.layers{l}.stepNo = [m.layers{l}.stepNo step];
for i = 1:length(m.layer_map.bigpos_hidden),
	step = step+1;
	for s = 1:length(m.layer_map.bigpos_hidden{i}),
		l = m.layer_map.bigpos_hidden{i}(s);
		m.layers{l}.stepNo = [m.layers{l}.stepNo step];
	end
end
step = step+1;
l = m.layer_map.bigpos_output;
m.layers{l}.stepNo = [m.layers{l}.stepNo step];
m.step_map.bigpos_fwd(2) = step;
% bigneg fwd pass
step = step+1;
m.step_map.bigneg_fwd(1) = step;
l = m.layer_map.bigneg_input;
m.layers{l}.stepNo = [m.layers{l}.stepNo step];
for i = 1:length(m.layer_map.bigneg_hidden),
	step = step+1;
	for s = 1:length(m.layer_map.bigneg_hidden{i}),
		l = m.layer_map.bigneg_hidden{i}(s);
		m.layers{l}.stepNo = [m.layers{l}.stepNo step];
	end
end
step = step+1;
l = m.layer_map.bigneg_output;
m.layers{l}.stepNo = [m.layers{l}.stepNo step];
m.step_map.bigneg_fwd(2) = step;

% fwd pass
step = step+1;
m.step_map.fwd(1) = step;
l = m.layer_map.input;
m.layers{l}.stepNo = [m.layers{l}.stepNo step];
for i = 1:length(m.layer_map.hidden),
	step = step+1;
	for s = 1:length(m.layer_map.hidden{i}),
		l = m.layer_map.hidden{i}(s);
		m.layers{l}.stepNo = [m.layers{l}.stepNo step];
	end
end
step = step+1;
l = m.layer_map.output;
m.layers{l}.stepNo = [m.layers{l}.stepNo step];
m.step_map.fwd(2) = step;

% bkwd pass
step = step+1;
m.step_map.bkwd(1) = step;
% l = m.layer_map.label;
% m.layers{l}.stepNo = [m.layers{l}.stepNo step];
% step = step+1;
l = m.layer_map.output;
m.layers{l}.stepNo = [m.layers{l}.stepNo step];
for i = length(m.layer_map.hidden):-1:1,
	step = step+1;
	for s = 1:length(m.layer_map.hidden{i}),
		l = m.layer_map.hidden{i}(s);
		m.layers{l}.stepNo = [m.layers{l}.stepNo step];
	end
end
m.step_map.bkwd(2) = step;

% gradient pass
step = step+1;
m.step_map.gradient = [step step+1];
for i = 1:length(m.layer_map.weight),
	for j = 1:length(m.layer_map.weight{i}),
		for k = 1:length(m.layer_map.weight{i}{j}),
			l = m.layer_map.weight{i}{j}(k);
			m.layers{l}.stepNo = [m.layers{l}.stepNo step step+1];
		end
		l = m.layer_map.bias{i}{j};
		m.layers{l}.stepNo = [m.layers{l}.stepNo step];
	end
end


return;
