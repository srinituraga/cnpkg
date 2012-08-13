function m = cnpkg_mknet;

%%% Data files %%%
% specify data files (can add more to these structs at a later time too)
m.data_info.training_file = '~sturaga/net_sets/ATLUM/bdry/cortex_layer1_39slices';
m.data_info.testing_file = '~sturaga/net_sets/ATLUM/bdry/cortex_layer1_39slices';

%%%%% DEBUG MODE %%%%%
m.params.debug=false;

% size of the minibatch (# of patches)
m.params.minibatch_size = 1;

% size of each individual minibatch sample on output end (in one dimension)
m.params.output_size=[1 1 1];

% minimax params
% m.params.graph_size = [300 300 1];
% m.params.constrained_minimax = true;
% m.params.nhood = mknhood2d(1);

% max # of total training iterations (not epochs)
m.params.maxIter = 2e6;
m.params.nIterPerEpoch = 1e4;
m.params.nEpochPerSave = 1e1;

m.params.zero_pad = true;

%%% NETWORK ARCHITECTURE %%%
[nhood1,nhood2] = mknhood_atlum(1,1);
m.params.input_units = 1;
m.params.output_units = size(nhood1,1);

% structure of each layer

l = 1;
m.params.layer{l}.nHid = {}; m.params.layer{l}.scales = {}; m.params.layer{l}.patchSz = {}; m.params.layer{l}.wPxSize = {};
m.params.layer{l}.nHid{end+1} = 24;
m.params.layer{l}.scales{end+1} = [1 1 1];
m.params.layer{l}.patchSz{end+1} = [[1 1]*5 3];
m.params.layer{l}.wPxSize{end+1} = [[1 1]*1 1];
m.params.layer{l}.nHid{end+1} = 16;
m.params.layer{l}.scales{end+1} = [1 1 1];
m.params.layer{l}.patchSz{end+1} = [[1 1]*5 5];
m.params.layer{l}.wPxSize{end+1} = [[1 1]*3 1];
m.params.layer{l}.nHid{end+1} = 12;
m.params.layer{l}.scales{end+1} = [1 1 1];
m.params.layer{l}.patchSz{end+1} = [[1 1]*5 5];
m.params.layer{l}.wPxSize{end+1} = [[1 1]*5 1];
m.params.layer{l}.nHid{end+1} = 12;
m.params.layer{l}.scales{end+1} = [1 1 1];
m.params.layer{l}.patchSz{end+1} = [[1 1]*5 5];
m.params.layer{l}.wPxSize{end+1} = [[1 1]*7 3];
m.params.layer{l}.nHid{end+1} = 12;
m.params.layer{l}.scales{end+1} = [1 1 1];
m.params.layer{l}.patchSz{end+1} = [[1 1]*5 5];
m.params.layer{l}.wPxSize{end+1} = [[1 1]*9 5];

% l = 2;
% m.params.layer{l}.nHid = {}; m.params.layer{l}.scales = {}; m.params.layer{l}.patchSz = {}; m.params.layer{l}.wPxSize = {};
% m.params.layer{l}.nHid{end+1} = 32;
% m.params.layer{l}.scales{end+1} = [1 1 1];
% m.params.layer{l}.patchSz{end+1} = [[1 1]*5 1];
% m.params.layer{l}.wPxSize{end+1} = [[1 1]*1 1];
% m.params.layer{l}.nHid{end+1} = 32;
% m.params.layer{l}.scales{end+1} = [1 1 1];
% m.params.layer{l}.patchSz{end+1} = [[1 1]*1 1];
% m.params.layer{l}.wPxSize{end+1} = [[1 1]*5 1];
% % m.params.layer{l}.nHid{end+1} = 16;
% % m.params.layer{l}.scales{end+1} = [1 1 1];
% % m.params.layer{l}.patchSz{end+1} = [[1 1]*1 1];
% % m.params.layer{l}.wPxSize{end+1} = [[1 1]*5 1];

% l = 3;
% m.params.layer{l}.nHid = {}; m.params.layer{l}.scales = {}; m.params.layer{l}.patchSz = {}; m.params.layer{l}.wPxSize = {};
% m.params.layer{l}.nHid{end+1} = 128;
% m.params.layer{l}.scales{end+1} = [1 1 1];
% m.params.layer{l}.patchSz{end+1} = [[1 1]*1 1];
% m.params.layer{l}.wPxSize{end+1} = [[1 1]*1 1];

m.params.num_layers=l+1;
m.params.layer{m.params.num_layers}.scales{1} = [1 1 1];
m.params.layer{m.params.num_layers}.patchSz{1} = [[1 1]*5 3];
m.params.layer{m.params.num_layers}.wPxSize{1} = [[1 1]*1 1];

m.globaleta = 2e-1;
m.binarythreshold = 0.5;
for l = 1:m.params.num_layers,
	for s = 1:length(m.params.layer{l}.scales),
		m.params.layer{l}.etaW{s} = 1;%(prod(m.params.layer{l}.scales{s}))^.3;
		m.params.layer{l}.etaB{s} = 1;
		m.params.layer{l}.initW{s} = 1e-0;
		m.params.layer{l}.initB{s} = 1e-0;
	end
end
for s = 1:length(m.params.layer{end}.scales),
	m.params.layer{end}.etaW{s} = 1e-1;
	m.params.layer{end}.etaB{s} = 1e-1;
end


%%% DIRECTORIES %%%
% where to save the network to? put a slash at the end
username = cnpkg_get_username;
m.params.save_directory=['~/saved_networks/'];

% global ID file
m.params.ID_file=['~/saved_networks/id_counter.txt'];


%%% VIEWING/SAVING %%%
% how many seconds between plotting status to the screen
m.params.view_interval=30;

% how much seconds between saving of network state
m.params.backup_interval=600;

% set up network based on these parameters
% initialize random number generators
rand('state',sum(100*clock));
randn('state',sum(100*clock));

% build the gpu cns model structure
m = cnpkg2_buildmodel(m);
m = cnpkg2_mapdim_layers_bkwd(m,m.params.output_size,m.params.minibatch_size*2);

% use outputSqSq
m.layers{m.layer_map.output}.type = 'outputSqSq';
m.layers{m.layer_map.output}.margin = .3;

% use conn label
m.layers{m.layer_map.label}.type = 'labelConn';
m.layers{m.layer_map.label}.nhood1 = nhood1;
m.layers{m.layer_map.label}.nhood2 = nhood2;

% assign an ID to this network
m.ID=cnpkg_get_id(m.params.ID_file);
ct=fix(clock);
m.date_created=[num2str(ct(2)),'-',num2str(ct(3)),'-',num2str(ct(4)),'-',num2str(ct(5))];
m.stats.iter=0;
m.stats.training_time=0;

m.params.layer
m
