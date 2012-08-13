function m = cnpkg_mknet;

%%% Data files %%%
% specify data files (can add more to these structs at a later time too)
m.data_info.training_file = '~sturaga/net_sets/cnpkg/BSD/train_color_boundary_mean';
m.data_info.testing_file = '';

%%%%% DEBUG MODE %%%%%
m.params.debug=false;

% size of the minibatch (# of patches)
m.params.minibatch_size = 1;

% size of each individual minibatch sample on output end (in one dimension)
m.params.output_size=[1 1 1];

% max # of total training iterations (not epochs)
m.params.maxIter=5e6;
m.params.nIterPerEpoch = 1e4;
m.params.nEpochPerSave = 1e1;


%%% NETWORK ARCHITECTURE %%%
m.params.input_units=3;
m.params.output_units = 1;
m.params.num_layers=2;

% structure of each layer
m.params.layer{1}.nHid = 20;
% m.params.layer{2}.nHid = 20;
% m.params.layer{3}.nHid = 20;
% m.params.layer{4}.nHid = 20;
% m.params.layer{5}.nHid = 20;

eta = 1e-0;
for l = 1:m.params.num_layers,
	m.params.layer{l}.patchSz = [[1 1]*7 1];
	m.params.layer{l}.etaW = eta;
	m.params.layer{l}.etaB = eta;
	m.params.layer{l}.initW = 1e-0;
	m.params.layer{l}.initB = 1e-0;
end
m.params.layer{l}.etaW = m.params.layer{l}.etaW * 1e-1;
m.params.layer{l}.etaB = m.params.layer{l}.etaB * 1e-1;


%%% DIRECTORIES %%%
% where to save the network to? put a slash at the end
username = cnpkg_get_username;
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

% assign an ID to this network
m.ID=cnpkg_get_id(m.params.ID_file);
ct=fix(clock);
m.date_created=[num2str(ct(2)),'-',num2str(ct(3)),'-',num2str(ct(4)),'-',num2str(ct(5))];
m.stats.iter=0;
m.stats.training_time=0;

% build the gpu cns model structure
m = cnpkg_buildmodel(m);

m.params.layer
m
