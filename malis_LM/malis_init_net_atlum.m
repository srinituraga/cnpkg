function m = malis_init_net(old_m)

old_m.stats = [];
m.old_m = old_m;
m.params = old_m.params;

%%% Data files %%%
% specify data files (can add more to these structs at a later time too)
m.data_info.training_files={...
'~sturaga/net_sets/ATLUM/malis/cortex_layer1_39slices'...
};
m.data_info.testing_files={...
};

%%%%% DEBUG MODE %%%%%
m.params.debug=false;

% size of the minibatch (# of patches)
m.params.minibatch_size = 5;

% size of each individual minibatch sample on output end (in one dimension)
m.params.output_size=[1 1 1];

% minimax parameters
m.params.graph_size = [1 1 1]*21;
m.params.nhood = mknhood2(1);
m.params.constrained_minimax = true;

% max # of total training iterations (not epochs)
m.params.maxIter=1e6;
m.params.nIterPerEpoch = 1e3;
m.params.nEpochPerSave = 1e1;


%%% NETWORK ARCHITECTURE %%%


%%% DIRECTORIES %%%
% where to save the network to? put a slash at the end
username = get_username;
m.params.save_directory=['/home/' username '/saved_networks/'];

% global ID file
m.params.ID_file=['/home/' username '/saved_networks/id_counter.txt'];


%%% VIEWING/SAVING %%%
m.params.backup_interval=600;

% set up network based on these parameters
% initialize random number generators
rand('state',sum(100*clock));
randn('state',sum(100*clock));

% assign an ID to this network
m.ID=get_id(m.params.ID_file);
ct=fix(clock);
m.date_created=[num2str(ct(2)),'-',num2str(ct(3)),'-',num2str(ct(4)),'-',num2str(ct(5))];
m.stats.iter=0;
m.stats.training_time=0;

% build the gpu cns model structure
m = malis_buildmodel(m);

% use sq-sq loss
m.layers{m.layer_map.output}.type = 'outputSqSq';
m.layers{m.layer_map.output}.margin = 0.3;

% copy in the old weights
for l = 1:length(m.old_m.layer_map.weight),
	for s = 1:length(m.old_m.layer_map.weight{l}),
		for w = 1:length(m.old_m.layer_map.weight{l}{s}),
			m.layers{m.layer_map.weight{l}{s}(w)}.val = ...
					m.old_m.layers{m.old_m.layer_map.weight{l}{s}(w)}.val;
% 			m.layers{m.layer_map.weight{l}{s}(w)}.eta = ...
% 					m.old_m.layers{m.old_m.layer_map.weight{l}{s}(w)}.eta;
		end
		m.layers{m.layer_map.bias{l}{s}}.val = ...
				m.old_m.layers{m.old_m.layer_map.bias{l}{s}}.val;
% 		m.layers{m.layer_map.bias{l}{s}}.eta = ...
% 				m.old_m.layers{m.old_m.layer_map.bias{l}{s}}.eta;
	end
end

m.params.layer
m
