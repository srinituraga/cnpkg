function m = cnpkg_mknet_addlayer_soft(old_m);
% cnpkg_mknet_addlayer_soft adds layer using paramters from existing layers
% Like Srini's cnpkg_mknet_addlayer, this makes a new convolutional neural
% network with one more hidden layer than old_m. UNLIKE
% cnpkg_mknet_addlayer, this function does NOT have any hard coded
% parameter values. Instead it uses parameter values from old_m. The newly
% created hidden layer has the same parameters (number of units, learning
% rate) as the hidden layer preceeding it. The learning rates of all other
% layers are preserved.

% copy in old m
old_m.stats = [];
m.old_m = old_m;

%%% Data files %%%
% specify data files (can add more to these structs at a later time too)
m.data_info.training_file = m.old_m.data_info.training_file;
m.data_info.testing_file = m.old_m.data_info.testing_file;

%%%%% DEBUG MODE %%%%%
m.params.debug=m.old_m.params.debug;

% size of the minibatch (# of patches)
m.params.minibatch_size = m.old_m.params.minibatch_size;

% size of each individual minibatch sample on output end (in one dimension)
m.params.output_size=m.old_m.params.output_size;

% % minimax params
% m.params.graph_size = [300 300 1];
% m.params.constrained_minimax = true;
% m.params.nhood = mknhood2d(1);

% max # of total training iterations (not epochs)
m.params.maxIter=m.old_m.params.maxIter;
m.params.nIterPerEpoch = m.old_m.params.nIterPerEpoch;
m.params.nEpochPerSave = m.old_m.params.nEpochPerSave;


%%% NETWORK ARCHITECTURE %%%
m.params.input_units = m.old_m.params.input_units;
m.params.output_units = m.old_m.params.output_units;
m.params.num_layers = m.old_m.params.num_layers+1;

% structure of each layer
for l = 1:m.old_m.params.num_layers-1,
	m.params.layer{l}.nHid = m.old_m.params.layer{l}.nHid;
end
m.params.layer{m.params.num_layers-1}.nHid = m.old_m.params.layer{l}.nHid;

for l = 1:m.old_m.params.num_layers,
    lold = max(1,l-1); %take etas from layer before, unless we are in the first layer
	m.params.layer{l}.patchSz = m.old_m.params.layer{lold}.patchSz;
	m.params.layer{l}.etaW = m.old_m.params.layer{lold}.etaW;
	m.params.layer{l}.etaB = m.old_m.params.layer{lold}.etaB;
	m.params.layer{l}.initW = 1e-0;
	m.params.layer{l}.initB = 1e-0;
end
m.params.layer{l+1}.patchSz = m.params.layer{l}.patchSz;
m.params.layer{l+1}.etaW = m.old_m.params.layer{end}.etaW;
m.params.layer{l+1}.etaB = m.old_m.params.layer{end}.etaB;
m.params.layer{l+1}.initW = 1e-0;
m.params.layer{l+1}.initB = 1e-0;


%%% DIRECTORIES %%%
% where to save the network to? put a slash at the end
username = cnpkg_get_username;
m.params.save_directory=['/Users/' username '/saved_networks/'];

% global ID file
m.params.ID_file=['/Users/' username '/saved_networks/id_counter.txt'];


%%% VIEWING/SAVING %%%
% how many seconds between plotting status to the screen
m.params.view_interval=m.old_m.params.view_interval;

% how much seconds between saving of network state
m.params.backup_interval=m.old_m.params.backup_interval;

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

% copy in the old weights
for l = 1:length(m.old_m.layer_map.weight)-1,
	m.layers{m.layer_map.weight(l)}.val = m.old_m.layers{m.old_m.layer_map.weight(l)}.val;
	m.layers{m.layer_map.bias(l)}.val = m.old_m.layers{m.old_m.layer_map.bias(l)}.val;
end


m.params.layer
m