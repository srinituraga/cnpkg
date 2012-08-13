function m = denoise_mknet

%%% Data files %%%
% specify data files (can add more to these structs at a later time too)
m.data_info.training_file = '/Users/stetner/data/denoising/set1_im_allnoise_1to40.mat';
m.data_info.testing_file  = '/Users/stetner/data/denoising/set1_im_25noise_1to40.mat';

%%%%% DEBUG MODE %%%%%
m.params.debug=false;

% size of the minibatch (# of patches divided by two)
m.params.minibatch_size = 3;

% size of each individual minibatch sample on output end (in one dimension)
m.params.output_size=[6 6 1];

% max # of total training iterations (not epochs)
m.params.nIterPerEpoch = 32768; % this is the max number of iterations that will fit in memory on my gpu
m.params.maxIter=30*5*40*321*481/216;% 30 epochs of 5 noise levels, 40 images, each image is 321x481, each minibatch is 6^3=216 pixels
m.params.nEpochPerSave = 1e1;


%%% NETWORK ARCHITECTURE %%%
m.params.input_units=1;
m.params.output_units = 1;
m.params.num_layers=2;

% structure of each layer
m.params.layer{1}.nHid = 48;
% m.params.layer{2}.nHid = 20;
% m.params.layer{3}.nHid = 20;
% m.params.layer{4}.nHid = 20;
% m.params.layer{5}.nHid = 20;

eta = 1e-1;
for l = 1:m.params.num_layers,
	m.params.layer{l}.patchSz = [[1 1]*5 1];
	m.params.layer{l}.etaW = eta;
	m.params.layer{l}.etaB = eta;
	m.params.layer{l}.initW = 1e-0;
	m.params.layer{l}.initB = 1e-0;
end
m.params.layer{l}.etaW = m.params.layer{l}.etaW * 1e-2;
m.params.layer{l}.etaB = m.params.layer{l}.etaB * 1e-2;
m.params.layer{l}.initW = 1e-0;
m.params.layer{l}.initB = 1e-0;


%%% DIRECTORIES %%%
% where to save the network to? put a slash at the end
username = cnpkg_get_username;
m.params.save_directory=['/Users/' username '/saved_networks/'];

% global ID file
m.params.ID_file=['/Users/' username '/saved_networks/id_counter.txt'];


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