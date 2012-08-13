function m = malis_mapdim_layers(m)

%% set up the minibatch index sizes
l = m.layer_map.minibatch_index;
m.layers{l}.val = 0;
m.layers{l}.size = {5, m.params.nIterPerEpoch, m.params.minibatch_size*2};

%% Setup the layer sizes for activity layers
% output layer
m.layers{m.layer_map.output}.size{5} = m.params.minibatch_size*2;
m.layers{m.layer_map.output}.val = 0;
m.layers{m.layer_map.output}.label = 0;
m.layers{m.layer_map.output}.mask = 0;
m.layers{m.layer_map.output}.loss = 0;
m.layers{m.layer_map.output}.sens = 0;
m.layers{m.layer_map.output}.labelblock{1} = [];
m.layers{m.layer_map.output}.maskblock{1} = [];
m = cns_mapdim(m, m.layer_map.output, 2, 'pixels', m.params.output_size(1));
m = cns_mapdim(m, m.layer_map.output, 3, 'pixels', m.params.output_size(2));
m = cns_mapdim(m, m.layer_map.output, 4, 'pixels', m.params.output_size(3));
% hidden layers
for i = length(m.layer_map.hidden):-1:1,
	m.layers{m.layer_map.hidden(i)}.size{5} = m.params.minibatch_size*2;
	m.layers{m.layer_map.hidden(i)}.val = 0;
	m.layers{m.layer_map.hidden(i)}.sens = 0;
    m = cns_mapdim(m, m.layer_map.hidden(i), 2, 'int-td', m.layer_map.hidden_next_layer(i), m.layers{m.layer_map.weight(i+1)}.size{2}, 1);
    m = cns_mapdim(m, m.layer_map.hidden(i), 3, 'int-td', m.layer_map.hidden_next_layer(i), m.layers{m.layer_map.weight(i+1)}.size{3}, 1);
    m = cns_mapdim(m, m.layer_map.hidden(i), 4, 'int-td', m.layer_map.hidden_next_layer(i), m.layers{m.layer_map.weight(i+1)}.size{4}, 1);
end
% input layer
m.layers{m.layer_map.input}.size{5} = m.params.minibatch_size*2;
m.layers{m.layer_map.input}.val = 0;
m.layers{m.layer_map.input}.inputblock{1} = [];
m = cns_mapdim(m, m.layer_map.input, 2, 'int-td', m.layer_map.hidden(1), m.layers{m.layer_map.weight(1)}.size{2}, 1);
m = cns_mapdim(m, m.layer_map.input, 3, 'int-td', m.layer_map.hidden(1), m.layers{m.layer_map.weight(1)}.size{3}, 1);
m = cns_mapdim(m, m.layer_map.input, 4, 'int-td', m.layer_map.hidden(1), m.layers{m.layer_map.weight(1)}.size{4}, 1);

%% Compute the "thickness" of the border lost due to valid convolutions
validBorder = (cell2mat(m.layers{m.layer_map.input}.size)-cell2mat(m.layers{m.layer_map.output}.size))/2;
validBorder = validBorder(2:4);
m.layers{m.layer_map.output}.offset = validBorder;

return
