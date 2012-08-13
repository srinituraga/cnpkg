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
%m.data_info.training_files={'~sturaga/net_sets/E2198/malis/e2006_e2198_plus_frmMattG'};
%m.data_info.training_files={'~sturaga/net_sets/E2198/malis/e2198_cellbodies'};
m.data_info.training_files={'~sturaga/net_sets/ISBI2012/challenge'};
m.data_info.testing_files={...
};

m.params.nDataBlock=50;
m.params.dataBlockSize=[512 512 1];
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
m.params.graph_size = [17 17 1];	% first start at this number
%m.params.graph_size = [25 25 1]; % then change to this number after training error decreases
m.params.nhood = mknhood2d(1);
m.params.constrained_minimax = false;

% max # of total training iterations (not epochs)
m.params.maxIter=1e6;
m.params.nIterPerEpoch = 1e3;
m.params.nEpochPerSave = 1e1;

% learning rate
m.globaleta = 0.1;

% margin
m.layers{m.layer_map.error}.param=.3;

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
