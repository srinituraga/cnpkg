function m = cnpkg3_mapdim_layers(m,input_size,minibatch_size)

%% set up the minibatch index sizes
l = m.layer_map.minibatch_index;
m.layers{l}.val = 0;
m.layers{l}.size = {5, m.params.nIterPerEpoch, minibatch_size};

%% Setup the layer sizes for activity layers
% input layer
l = m.layer_map.input;
m.layers{l}.size{5} = minibatch_size;
m.layers{l}.val = 0;
m.inputblock = {[]};
m.layers{l}.y_space = 1; m.layers{l}.x_space = 1; m.layers{l}.d_space = 1;
m.layers{l}.y_start = 0; m.layers{l}.x_start = 0; m.layers{l}.d_start = 0;
m.layers{l}.size{2} = input_size(1); m.layers{l}.size{3} = input_size(2); m.layers{l}.size{4} = input_size(3);
% hidden layers
for i = 1:length(m.layer_map.weight)-1,
	for s = 1:length(m.layer_map.weight{i}),
		l = m.layer_map.hidden{i}(s);
		m.layers{l}.size{5} = minibatch_size;
		m.layers{l}.val = 0;
		m.layers{l}.sens = 0;
		m.layers{l}.y_space = m.params.layer{i}.scales{s}(1);
		m.layers{l}.x_space = m.params.layer{i}.scales{s}(2);
		m.layers{l}.d_space = m.params.layer{i}.scales{s}(3);
		m.layers{l}.y_start = -inf; m.layers{l}.x_start = -inf; m.layers{l}.d_start = -inf;
		m.layers{l}.size{2} = inf; m.layers{l}.size{3} = inf; m.layers{l}.size{4} = inf;
		for sp = 1:length(m.layer_map.weight{i}{s}),
			lw = m.layer_map.weight{i}{s}(sp);
			lp = m.layers{lw}.zp;

			offset = floor(m.layers{lw}.size{2}/2);
			m.layers{l}.y_start = max(m.layers{l}.y_start,m.layers{lp}.y_start+offset*m.layers{lp}.y_space);
			m.layers{l}.size{2} = min(m.layers{l}.size{2}, ...
							floor((m.layers{lp}.size{2}-2*offset-1)*m.layers{lp}.y_space/m.layers{l}.y_space)+1);
			offset = floor(m.layers{lw}.size{3}/2);
			m.layers{l}.x_start = max(m.layers{l}.x_start,m.layers{lp}.x_start+offset*m.layers{lp}.x_space);
			m.layers{l}.size{3} = min(m.layers{l}.size{3}, ...
							floor((m.layers{lp}.size{3}-2*offset-1)*m.layers{lp}.x_space/m.layers{l}.x_space)+1);
			offset = floor(m.layers{lw}.size{4}/2);
			m.layers{l}.d_start = max(m.layers{l}.d_start,m.layers{lp}.d_start+offset*m.layers{lp}.d_space);
			m.layers{l}.size{4} = min(m.layers{l}.size{4}, ...
							floor((m.layers{lp}.size{4}-2*offset-1)*m.layers{lp}.d_space/m.layers{l}.d_space)+1);

		end
	end
end
% output layer
m.layers{m.layer_map.output}.size{5} = minibatch_size;
m.layers{m.layer_map.output}.val = 0;
m.layers{m.layer_map.output}.loss = 0;
m.layers{m.layer_map.output}.classerr = 0;
m.layers{m.layer_map.output}.sens = 0;
m.layers{m.layer_map.label}.size{5} = minibatch_size;
m.layers{m.layer_map.label}.val = 0;
m.layers{m.layer_map.label}.mask = 0;
m.labelblock{1} = [];
m.maskblock{1} = [];

l = m.layer_map.output;
m.layers{l}.y_space = 1; m.layers{l}.x_space = 1; m.layers{l}.d_space = 1;
m.layers{l}.y_start = -inf; m.layers{l}.x_start = -inf; m.layers{l}.d_start = -inf;
m.layers{l}.size{2} = inf; m.layers{l}.size{3} = inf; m.layers{l}.size{4} = inf;
for sp = 1:length(m.layer_map.weight{end}{1}),
	lw = m.layer_map.weight{end}{1}(sp);
	lp = m.layers{lw}.zp;

	offset = floor(m.layers{lw}.size{2}/2);
	m.layers{l}.y_start = max(m.layers{l}.y_start,m.layers{lp}.y_start+offset*m.layers{lp}.y_space);
	m.layers{l}.size{2} = min(m.layers{l}.size{2}, ...
					floor((m.layers{lp}.size{2}-2*offset-1)*m.layers{lp}.y_space/m.layers{l}.y_space)+1);
	offset = floor(m.layers{lw}.size{3}/2);
	m.layers{l}.x_start = max(m.layers{l}.x_start,m.layers{lp}.x_start+offset*m.layers{lp}.x_space);
	m.layers{l}.size{3} = min(m.layers{l}.size{3}, ...
					floor((m.layers{lp}.size{3}-2*offset-1)*m.layers{lp}.x_space/m.layers{l}.x_space)+1);
	offset = floor(m.layers{lw}.size{4}/2);
	m.layers{l}.d_start = max(m.layers{l}.d_start,m.layers{lp}.d_start+offset*m.layers{lp}.d_space);
	m.layers{l}.size{4} = min(m.layers{l}.size{4}, ...
					floor((m.layers{lp}.size{4}-2*offset-1)*m.layers{lp}.d_space/m.layers{l}.d_space)+1);
end
m = cns_mapdim(m, m.layer_map.label, 2, 'copy', m.layer_map.output);
m = cns_mapdim(m, m.layer_map.label, 3, 'copy', m.layer_map.output);
m = cns_mapdim(m, m.layer_map.label, 4, 'copy', m.layer_map.output);

%% Compute the "thickness" of the border lost due to valid convolutions
% shrinkSize = (cell2mat(m.layers{m.layer_map.input}.size)-cell2mat(m.layers{m.layer_map.output}.size))/2;
% shrinkSize = shrinkSize(2:4);
% m.offset = shrinkSize;
m.offset(1) = m.layers{m.layer_map.output}.x_start-m.layers{m.layer_map.input}.x_start;
m.offset(2) = m.layers{m.layer_map.output}.y_start-m.layers{m.layer_map.input}.y_start;
m.offset(3) = m.layers{m.layer_map.output}.d_start-m.layers{m.layer_map.input}.d_start;

m.totalBorder = cell2mat(m.layers{m.layer_map.input}.size(2:4)) - cell2mat(m.layers{m.layer_map.output}.size(2:4));
m.leftBorder = m.offset;
m.rightBorder = m.totalBorder - m.leftBorder;

for l = 1:length(m.layers),
	switch m.layers{l}.type,
	case 'input',
		m.layers{l}.val = 0;
		m.inputblock = {[]};
	case 'label',
		m.layers{l}.val = 0;
		m.labelblock = {[]};
		m.maskblock = {[]};
	end
end

return
