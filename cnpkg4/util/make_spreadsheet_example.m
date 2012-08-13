clear lbl
clear v
prep =[];


prep{1} = 'tend = find(m.stats.loss(end:-1:1),1,''first'');';
prep{2} = 'tend = 1+length(m.stats.loss)-tend;';
% prep{end+1} = 'if m.ID == 1100036; figure;plot(m.stats.loss); end;';


lbl{1} = 'ID';
v{1} = 'm.ID';

lbl{end+1} = 'iters';
v{end+1} = 'm.stats.iter';


lbl{end+1} = 'dataset';
v{end+1} = 'm.data_info.training_file';

lbl{end+1} = 'num_layers';
v{end+1} = 'm.params.num_layers';

lbl{end+1} = 'L1scl1';
v{end+1} = 'm.params.layer{1}.scales{2}';

lbl{end+1} = 'L1scl2';
v{end+1} = 'm.params.layer{1}.scales{3}';

lbl{end+1} = 'L2scl1';
v{end+1} = 'm.params.layer{2}.scales{2}';

lbl{end+1} = 'nhid';
v{end+1} = 'sum(m.params.layer{1}.nHid{1})';

lbl{end+1} = 'eta';
v{end+1} = 'm.globaleta';

lbl{end+1} = 'perf';
v{end+1} = 'mean(m.stats.loss(tend-49:tend))';

lbl{end+1} = 'finlayeretaratio';
v{end+1} = 'm.params.layer{end}.etaW{end}/m.params.layer{1}.etaW{end}';

lbl{end+1} = 'Fsize';
v{end+1} = 'm.params.layer{1}.patchSz{1}';

lbl{end+1} = 'fov';
v{end+1} = 'm.totalBorder';



a = browseNetworks('~/cns_files/nets/atum_2d/',v, lbl, prep);

struct2tdtxt('~/cns_files/atum2d_info.txt',a);