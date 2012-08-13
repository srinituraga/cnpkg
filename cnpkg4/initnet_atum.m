function m = initnet(m,old_m);

if exist('old_m','var') && ~isempty(old_m),
    m.old_m = old_m;
end
m.package = 'cnpkg4';
m.independent = true;

%%% Data files %%%
% specify data files (can add more to these structs at a later time too)
m.data_info.training_file = '~sturaga/net_sets/ATLUM/bdry/cortex_layer1_layer5';
m.data_info.testing_file = '~sturaga/net_sets/ATLUM/bdry/cortex_layer1_layer5';

%%%%% DEBUG MODE %%%%%
m.params.debug=false;

% size of the minibatch (# of patches)
m.params.minibatch_size = 1;

% size of each individual minibatch sample on output end (in one dimension)
m.params.output_size=[11 11 1]*1;

% minimax params
% m.params.graph_size = [300 300 1];
% m.params.constrained_minimax = true;
% m.params.nhood = mknhood2d(1);

% max # of total training iterations (not epochs)
m.params.maxIter = 2e6;
m.params.nIterPerEpoch = 1e3;
m.params.nEpochPerSave = 1e1;

m.params.zero_pad = false;

%%% NETWORK ARCHITECTURE %%%
[nhood1,nhood2] = mknhood_atlum(1,1);
m.params.input_units = 1;
m.params.output_units = size(nhood1,1);

% % use SqSq error
m.layers{m.layer_map.error}.kernel = {'SqSq'};
m.layers{m.layer_map.error}.param = .3;

% use conn label
m.layers{m.layer_map.label}.type = 'labelConn';
m.layers{m.layer_map.label}.nhood1 = nhood1;
m.layers{m.layer_map.label}.nhood2 = nhood2;


% learning rate
m.globaleta = 2e-1;
m.binarythreshold = 0.5;


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

% assign an ID to this network
m = assignID(m);

m.stats.iter=0;
