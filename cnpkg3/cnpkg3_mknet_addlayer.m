function m = cnpkg_mknet_addlayer(old_m);

% copy in old m
old_m.stats = [];
m.old_m = old_m;

%%% Data files %%%
% specify data files (can add more to these structs at a later time too)
m.data_info.training_file = '~sturaga/net_sets/cnpkg/BSD/train_color_boundary_mean';
m.data_info.testing_file = '~sturaga/net_sets/cnpkg/BSD/test_color_boundary_mean';

%%%%% DEBUG MODE %%%%%
m.params.debug=false;

% size of the minibatch (# of patches)
m.params.minibatch_size = 1;

% size of each individual minibatch sample on output end (in one dimension)
m.params.output_size=[1 1 1];

% % minimax params
% m.params.graph_size = [300 300 1];
% m.params.constrained_minimax = true;
% m.params.nhood = mknhood2d(1);

% max # of total training iterations (not epochs)
m.params.maxIter=1e7;
m.params.nIterPerEpoch = 1e4;
m.params.nEpochPerSave = 1e1;


%%% NETWORK ARCHITECTURE %%%
m.params.input_units = m.old_m.params.input_units;
m.params.output_units = m.old_m.params.output_units;
m.params.num_layers = m.old_m.params.num_layers+1;

% structure of each layer
m.globaleta = m.old_m.globaleta;
m.params.layer(1:m.params.num_layers-2) = m.old_m.params.layer(1:m.params.num_layers-2);
% new layer
l = m.params.num_layers-1;
m.params.layer{l}.nHid = {}; m.params.layer{l}.scales = {}; m.params.layer{l}.patchSz = {};
m.params.layer{l}.nHid{end+1} = 64;
m.params.layer{l}.scales{end+1} = [1 1 1];
m.params.layer{l}.patchSz{end+1} = [[1 1]*1 1];
% m.params.layer{l}.nHid{end+1} = 20;
% m.params.layer{l}.scales{end+1} = [3 3 1];
% m.params.layer{l}.patchSz{end+1} = [[1 1]*1 1];
% m.params.layer{l}.nHid{end+1} = 20;
% m.params.layer{l}.scales{end+1} = [5 5 1];
% m.params.layer{l}.patchSz{end+1} = [[1 1]*1 1];

l = m.params.num_layers;
m.params.layer{l}.scales{1} = [1 1 1];
m.params.layer{l}.patchSz{1} = [[1 1] 1];
%m.params.layer(m.params.num_layers-1) = m.params.layer(m.params.num_layers-2);
%m.params.layer(m.params.num_layers) = m.old_m.params.layer(end);

m.globaleta = 5e-2;
m.binarythreshold = 0.86;
for l = 1:m.params.num_layers,
	for s = 1:length(m.params.layer{l}.scales),
		m.params.layer{l}.etaW{s} = 1e-1;%sqrt(prod(m.params.layer{l}.scales{s}));
		m.params.layer{l}.etaB{s} = 1e-1;
		m.params.layer{l}.initW{s} = 1e-1;
		m.params.layer{l}.initB{s} = 1e-0;
	end
end
for s = 1:length(m.params.layer{end}.scales),
	m.params.layer{end}.etaW{s} = 1e-0;
	m.params.layer{end}.etaB{s} = 1e-0;
end

%%% DIRECTORIES %%%
% where to save the network to? put a slash at the end
username = get_username;
m.params.save_directory=['/home/' username '/saved_networks/'];

% global ID file
m.params.ID_file=['/home/' username '/saved_networks/id_counter.txt'];


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
m = cnpkg3_buildmodel(m);
m = cnpkg3_mapdim_layers_bkwd(m,m.params.output_size,m.params.minibatch_size*2);

% assign an ID to this network
m.ID=cnpkg_get_id(m.params.ID_file);
ct=fix(clock);
m.date_created=[num2str(ct(2)),'-',num2str(ct(3)),'-',num2str(ct(4)),'-',num2str(ct(5))];
m.stats.iter=0;
m.stats.training_time=0;

% copy in the old weights
for l = 1:length(m.old_m.layer_map.weight)-1,
	for s = 1:length(m.old_m.layer_map.weight{l}),
		for w = 1:length(m.old_m.layer_map.weight{l}{s}),
			m.layers{m.layer_map.weight{l}{s}(w)}.val = ...
					m.old_m.layers{m.old_m.layer_map.weight{l}{s}(w)}.val;
% 			m.layers{m.layer_map.weight{l}{s}(w)}.eta = 5e-2;
		end
		m.layers{m.layer_map.bias{l}{s}}.val = ...
				m.old_m.layers{m.old_m.layer_map.bias{l}{s}}.val;
% 		m.layers{m.layer_map.bias{l}{s}}.eta = 5e-2;
	end
end


m.params.layer
m
