function m = cnpkg2_mapdim_layers(m,output_size,minibatch_size)

%% set up the minibatch index sizes
l = m.layer_map.minibatch_index;
m.layers{l}.val = 0;
m.layers{l}.size = {5, m.params.nIterPerEpoch, minibatch_size};

%% Setup the layer sizes for activity layers
% output layer
m.layers{m.layer_map.output}.size{5} = minibatch_size;
m.layers{m.layer_map.output}.val = 0;
m.layers{m.layer_map.output}.loss = 0;
m.layers{m.layer_map.output}.classerr = 0;
m.layers{m.layer_map.output}.sens = 0;
m.layers{m.layer_map.label}.size{5} = minibatch_size;
m.layers{m.layer_map.label}.val = 0;
m.layers{m.layer_map.label}.mask = 0;
m.layers{m.layer_map.label}.labelblock{1} = [];
m.layers{m.layer_map.label}.maskblock{1} = [];

m.layers{m.layer_map.output}.size{2} = output_size(1);
m.layers{m.layer_map.output}.y_start = 0;
m.layers{m.layer_map.output}.y_space = 1;
m.layers{m.layer_map.output}.size{3} = output_size(2);
m.layers{m.layer_map.output}.x_start = 0;
m.layers{m.layer_map.output}.x_space = 1;
m.layers{m.layer_map.output}.size{4} = output_size(3);
m.layers{m.layer_map.output}.d_start = 0;
m.layers{m.layer_map.output}.d_space = 1;
m = cns_mapdim(m, m.layer_map.label, 2, 'copy', m.layer_map.output);
m = cns_mapdim(m, m.layer_map.label, 3, 'copy', m.layer_map.output);
m = cns_mapdim(m, m.layer_map.label, 4, 'copy', m.layer_map.output);
% hidden layers
for i = length(m.layer_map.hidden):-1:1,
	for s = 1:length(m.layer_map.hidden{i}),
		l = m.layer_map.hidden{i}(s);
		m.layers{l}.size{5} = minibatch_size;
		m.layers{l}.val = 0;
		m.layers{l}.sens = 0;
		m.layers{l}.y_space = m.params.layer{i}.scales{s}(1);
		m.layers{l}.x_space = m.params.layer{i}.scales{s}(2);
		m.layers{l}.d_space = m.params.layer{i}.scales{s}(3);
		m.layers{l}.y_start = inf; m.layers{l}.x_start = inf; m.layers{l}.d_start = inf;
		m.layers{l}.size{2} = -inf; m.layers{l}.size{3} = -inf; m.layers{l}.size{4} = -inf;
		for sn = 1:length(m.layer_map.weight_out{i}{s}),
			lwn = m.layer_map.weight_out{i}{s}(sn);
			ln = m.layers{lwn}.zn;

			offset = floor(m.layers{lwn}.size{2}/2);
			m.layers{l}.y_start = min(m.layers{l}.y_start,m.layers{ln}.y_start-offset*m.layers{l}.y_space);
			m.layers{l}.size{2} = max(m.layers{l}.size{2}, ...
							round((m.layers{ln}.size{2}-1)*m.layers{ln}.y_space/m.layers{l}.y_space)+1+2*offset);
			offset = floor(m.layers{lwn}.size{3}/2);
			m.layers{l}.x_start = min(m.layers{l}.x_start,m.layers{ln}.x_start-offset*m.layers{l}.x_space);
			m.layers{l}.size{3} = max(m.layers{l}.size{3}, ...
							round((m.layers{ln}.size{3}-1)*m.layers{ln}.x_space/m.layers{l}.x_space)+1+2*offset);
			offset = floor(m.layers{lwn}.size{4}/2);
			m.layers{l}.d_start = min(m.layers{l}.d_start,m.layers{ln}.d_start-offset*m.layers{l}.d_space);
			m.layers{l}.size{4} = max(m.layers{l}.size{4}, ...
							round((m.layers{ln}.size{4}-1)*m.layers{ln}.d_space/m.layers{l}.d_space)+1+2*offset);
		end
	end
end
% input layer
l = m.layer_map.input;
m.layers{l}.size{5} = minibatch_size;
m.layers{l}.val = 0;
m.layers{l}.inputblock{1} = [];
m.layers{l}.y_space = 1; m.layers{l}.x_space = 1; m.layers{l}.d_space = 1;
m.layers{l}.y_start = inf; m.layers{l}.x_start = inf; m.layers{l}.d_start = inf;
m.layers{l}.size{2} = -inf; m.layers{l}.size{3} = -inf; m.layers{l}.size{4} = -inf;
for s = 1:length(m.layer_map.weight{1}),
	lwn = m.layer_map.weight{i}{s};
	ln = m.layers{lwn}.zn;

	offset = floor(m.layers{lwn}.size{2}/2);
	m.layers{l}.y_start = min(m.layers{l}.y_start,m.layers{ln}.y_start-offset*m.layers{l}.y_space);
	m.layers{l}.size{2} = max(m.layers{l}.size{2}, ...
					ceil(m.layers{ln}.size{2}*m.layers{ln}.y_space/m.layers{l}.y_space) + 2*offset);
	offset = floor(m.layers{lwn}.size{3}/2);
	m.layers{l}.x_start = min(m.layers{l}.x_start,m.layers{ln}.x_start-offset*m.layers{l}.x_space);
	m.layers{l}.size{3} = max(m.layers{l}.size{3}, ...
					ceil(m.layers{ln}.size{3}*m.layers{ln}.x_space/m.layers{l}.x_space) + 2*offset);
	offset = floor(m.layers{lwn}.size{4}/2);
	m.layers{l}.d_start = min(m.layers{l}.d_start,m.layers{ln}.d_start-offset*m.layers{l}.d_space);
	m.layers{l}.size{4} = max(m.layers{l}.size{4}, ...
					ceil(m.layers{ln}.size{4}*m.layers{ln}.d_space/m.layers{l}.d_space) + 2*offset);
end

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

return
