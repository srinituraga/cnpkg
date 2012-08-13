% similar to cnpkg_mknet_addlayer_soft.m; probably works but not fully debugged yet!

function m = cnpkg_mknet_addlayer(old_m);

% copy in old m
old_m.stats = [];
m.old_m = old_m;

m.data_info = m.old_m.data_info;
m.params = m.old_m.params;

%%% NETWORK ARCHITECTURE %%%
m.params.num_layers = m.old_m.params.num_layers+1;
for l = m.params.num_layers+(-1:0);
	m.params.layer{l} = m.old_m.params.layer{l-1};
end

if isfield(m.old_m,'globaleta'), m.globaleta = m.old_m.globaleta; end
if isfield(m.old_m,'binarythreshold'), m.binarythreshold = m.old_m.binarythreshold; end

% build the gpu cns model structure
m = cnpkg3_buildmodel(m);
m = cnpkg3_mapdim_layers_bkwd(m,m.params.output_size,m.params.minibatch_size*2);

%%% Copy in old weights
% copy in the old weights
for l = 1:length(m.old_m.layer_map.weight)-1,
	for s = 1:length(m.old_m.layer_map.weight{l}),
		for w = 1:length(m.old_m.layer_map.weight{l}{s}),
			m.layers{m.layer_map.weight{l}{s}(w)}.val = m.old_m.layers{m.old_m.layer_map.weight{l}{s}(w)}.val;
		end
		m.layers{m.layer_map.bias{l}{s}}.val = m.old_m.layers{m.old_m.layer_map.bias{l}{s}}.val;
	end
end

m.layers{m.layer_map.output}.type = m.old_m.layers{m.old_m.layer_map.output}.type;
if(isfield(m.old_m.layers{m.old_m.layer_map.output}, 'margin'))
	m.layers{m.layer_map.output}.margin = m.old_m.layers{m.old_m.layer_map.output}.margin;
end

m.layers{m.layer_map.label}.type = m.old_m.layers{m.old_m.layer_map.label}.type;
if isequal(m.old_m.layers{m.old_m.layer_map.label}.type, 'labelConn'),
	m.layers{m.layer_map.label}.nhood1 = m.old_m.layers{m.old_m.layer_map.label}.nhood1;
	m.layers{m.layer_map.label}.nhood2 = m.old_m.layers{m.old_m.layer_map.label}.nhood2;
end


%%% DIRECTORIES %%%
% where to save the network to? put a slash at the end
username = get_username;
m.params.save_directory=['/home/' username '/saved_networks/'];

% global ID file
m.params.ID_file=['/home/' username '/saved_networks/id_counter.txt'];

% assign an ID to this network
m.ID=cnpkg_get_id(m.params.ID_file);
ct=fix(clock);
m.date_created=[num2str(ct(2)),'-',num2str(ct(3)),'-',num2str(ct(4)),'-',num2str(ct(5))];
m.stats.iter=0;
m.stats.training_time=0;

m.params.layer
m
