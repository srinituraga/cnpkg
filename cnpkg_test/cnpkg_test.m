p = struct;
p.fCount  = [1 10 10 10 10 10 3];
p.fSize   = [0 5 5 5 5 5 5];
p.fDepth  = [0 5 5 5 5 5 5];
p.eta     = [0 1 1 1 1 1 1] * 1e-3;
p.outSize = [6 6 6];
p.iterations = 2;
m = cnpkg_buildmodel(p);

cns('init', m, 'gpu', 'mean');

input = rand(1, 30, 30, 30);
correct = rand(3, 6, 6, 6);

tic;
cns('set', 1, 'val', input);
cns('set', numel(m.layers), 'correct', correct);
% cns('step', 1, 6);
cns('run');
toc;

cns('done');