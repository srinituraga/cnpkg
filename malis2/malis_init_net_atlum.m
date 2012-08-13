function m = malis_init_net(m)

m.package = 'cnpkg4';
m.independent = true;

old_m = m;
old_m.stats = [];
m.old_m = old_m;
if isfield(m,'params'),
    m.params = old_m.params;
end

%%% Data files %%%
% specify data files (can add more to these structs at a later time too)
%m.data_info.training_files={'~sturaga/net_sets/ATLUM/malis/cortex_layer1_layer5_flps'};
m.data_info.training_files={'~sturaga/net_sets/ATLUM/malis/cortex_layer5_full'};
m.data_info.testing_files={...
};

m.params.nDataBlock=10;
m.params.dataBlockSize=[512 512 100];
m.params.dataBlockTransformFlp=[1 1 0];
m.params.dataBlockTransformPrmt=1;
m.params.nEpochPerDataBlock=1;


%%%%% DEBUG MODE %%%%%
m.params.debug=false;

%%%%%% PARAMETERS %%%%%
% size of the minibatch (# of patches)
m.params.minibatch_size = 1;

% size of each individual minibatch sample on output end (in one dimension)
m.params.output_size=[1 1 1];

% minimax parameters
m.params.graph_size = [1 1 1]*21;
m.params.nhood = mknhood2(1);
m.params.constrained_minimax = false;

% max # of total training iterations (not epochs)
m.params.maxIter=1e6;
m.params.nIterPerEpoch = 1e3;
m.params.nEpochPerSave = 1e1;

% learning rate
m.globaleta = 5;

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
m = assignID(m);
m.stats.iter=0;
m.stats.epoch=0;
m.stats.training_time=0;
