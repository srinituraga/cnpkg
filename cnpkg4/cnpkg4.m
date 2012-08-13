classdef cnpkg4 < cns_package
methods (Static)

%***********************************************************
function m = ComputeNodeTopoOrder(m)
% sort all the image nodes in topological order.
% this ensures that future operations like mapdim to work correctly

m.nodes = []; m.fwdPassNodes = []; m.bkwdPassNodes = [];
for l = 1:length(m.layers)
    if any(ismember(superclasses([m.package '_' m.layers{l}.type]),[m.package '_node'])),
        m.nodes(end+1) = l;
        deps = cns_call(m, l, 'GetDependencies');
        DepMat(1:length(deps),l) = deps(:);

        type = [m.package '_' m.layers{l}.type];
        classes = superclasses(type); classes{end+1} = type;
        if ismember([m.package '_computed'], classes) || ismember([m.package '_input'], classes),
            m.fwdPassNodes(end+1) = l;
        end
        if ismember([m.package '_sens'], classes) || ismember([m.package '_label'], classes),
            m.bkwdPassNodes(end+1) = l;
        end

    end
end
DepMat = DepMat(:,m.nodes);

assigned = repmat(false, 1, numel(m.nodes)); % step number
NodeTopoOrder = repmat(-1, 1, numel(m.nodes)); % step number
n = 0;
while true
    % find all unassigned nodes with no unsatisfied dependencies
    f = find(~assigned & all(DepMat == 0, 1));
    if isempty(f), break; end

    % assign step no
    NodeTopoOrder(n+(1:length(f))) = m.nodes(f);
    n = n+length(f);

    % say that the dependencies have now been satisfied
    DepMat(ismember(DepMat, m.nodes(f))) = 0;
    assigned(f) = true;
end

if any(NodeTopoOrder < 0), error('there is a dependency loop'); end

m.nodes = NodeTopoOrder;
m.fwdPassNodes = m.nodes(ismember(m.nodes,m.fwdPassNodes));
m.bkwdPassNodes = m.nodes(ismember(m.nodes,m.bkwdPassNodes));

end


%***********************************************************
function m = MapDimFromOutput(m,outputSize,nMiniBatch)
% setup the sizes of all the nodes
% assumes the node DAG is topologically sorted

dims = ['y', 'x', 'd'];

m = cnpkg4.ComputeNodeTopoOrder(m);

for l = m.nodes,
    m.layers{l}.size(2:4) = {0,0,0};
    if any(l == m.layer_map.output) || strcmp(m.layers{l}.name,'output'),
        for k=1:3,
            m.layers{l}.size{1+k} = outputSize(k);
            m.layers{l}.([dims(k) '_start']) = 0;
        end
        m.layers{l}.size{5} = nMiniBatch;
    end
end

for l = [m.fwdPassNodes(end:-1:1) m.bkwdPassNodes],
    typemethods = methods([m.package '_' m.layers{l}.type]);
    if ismember('MapDimBkwd', typemethods),
        m = cns_call(m, l, 'MapDimBkwd');
    end
end

for k = 1:3,
    m.totalBorder(k) = m.layers{m.layer_map.input}.size{k+1} ...
                        - m.layers{m.layer_map.output}.size{k+1};
    m.offset(k) = m.layers{m.layer_map.output}.([dims(k) '_start']) ...
                        - m.layers{m.layer_map.input}.([dims(k) '_start']);
end
m.leftBorder = m.offset;
m.rightBorder = m.totalBorder - m.leftBorder;

if any(m.leftBorder < 0) || any(m.rightBorder < 0),
    error('Borders are all wrong!!')
end

end


%***********************************************************
function m = MapDimFromInput(m,inputSize,nMiniBatch)
% setup the sizes of all the nodes
% assumes the node DAG is topologically sorted

if ~isfield(m,'NodeTopoOrder'),
    m = cnpkg4.ComputeNodeTopoOrder(m);
end

for l = m.NodeTopoOrder,
    type = [m.package '_' m.layers{l}.type];
    classes = superclasses(type); classes{end+1} = type;
    if ismember([m.package '_input'], classes),
    % setup the size of the input node
        for k=1:3,
            m.layers{l}.size{1+k} = inputSize(k);
        end
        m.layers{l}.size{5} = nMiniBatch;
    end
end

for l = m.NodeTopoOrder,
    typemethods = methods([m.package '_' m.layers{l}.type]);
    if ismember('MapDimFwd', typemethods),
        m = cns_call(m, l, 'MapDimFwd');
    end
end

for l = m.NodeTopoOrder(end:-1:1),
    typemethods = methods([m.package '_' m.layers{l}.type]);
    if ismember('MapDimBkwd', typemethods),
        m = cns_call(m, l, 'MapDimBkwd');
    end
end

for k = 1:3,
    m.totalBorder(k) = m.layers{m.layer_map.input}.size{k+1} ...
                        - m.layers{m.layer_map.output}.size{k+1};
    m.offset(k) = m.layers{m.layer_map.output}.([dims(k) '_start']) ...
                        - m.layers{m.layer_map.input}.([dims(k) '_start']);
end
m.leftBorder = m.offset;
m.rightBorder = m.totalBorder - m.leftBorder;

end


%***********************************************************
function [m,lastStep] = SetupStepNo(m,firstStep)
% adapted from CNS_SETUPSTEPNOS

DepMat = repmat(0, 0, numel(m.layers)); % dependencies

for l = 1:length(m.layers)
    deps = cns_call(m, l, 'GetDependencies');
    deps = unique(deps);
    DepMat(1:length(deps),l) = deps(:);
end

steps = repmat(-1, 1, numel(m.layers)); % step number
n = firstStep;

while true

    % find all unassigned nodes with no unsatisfied dependencies
    f = find((steps < 0) & all(DepMat == 0, 1));
    if isempty(f), break; end

    % assign step no
    steps(f) = n;
    n = n+1;

    % say that the dependencies have now been satisfied
    DepMat(ismember(DepMat, f)) = 0;

end

if any(steps < 0), error('there is a dependency loop'); end

for l = 1:length(m.layers)
    type = [m.package '_' m.layers{l}.type];
    classes = superclasses(type); classes{end+1} = type;
    if steps(l) == 0 || strcmp(m.layers{l}.type,'index') || (ismember([m.package '_weight'], classes) && m.layers{l}.eta == 0) || (ismember([m.package '_bias'], classes) && m.layers{l}.eta == 0),
        m.layers{l}.stepNo = [];
    else
        m.layers{l}.stepNo = steps(l);
    end
end
lastStep = max(steps);

% add another step for the weights to do apply the update and the constraints
lastStep = lastStep + 1;
for l = 1:length(m.layers)
    type = [m.package '_' m.layers{l}.type];
    classes = superclasses(type); classes{end+1} = type;
    if ismember([m.package '_weight'], classes) && m.layers{l}.eta > 0,
        m.layers{l}.stepNo(end+1) = lastStep;
        m.layers{l}.kernel = {'','constrain'};
    end
end

end


%***********************************************************
function m = SetupForTesting(m)

oldNumbers = 1:length(m.layers);
pruneNodes = [];
for l = 1:length(m.layers),
    type = [m.package '_' m.layers{l}.type];
    classes = superclasses(type); classes{end+1} = type;
    if ismember([m.package '_error'], classes) || ismember([m.package '_sens'], classes) || ismember([m.package '_label'], classes) || ismember([m.package '_index'], classes),
        pruneNodes(end+1) = l;
    elseif ismember([m.package '_computed'], classes),
        m.layers{l}.val = 0;
    end
    if ismember([m.package '_weight'], classes) && isfield(m.layers{l},'kernel'),
        m.layers{l} = rmfield(m.layers{l},'kernel');
    end
end

oldNumbers(pruneNodes) = [];

% apply the renumbering
m.layers(pruneNodes) = [];
for l = 1:length(m.layers),
    m = cns_call(m,l,'RenumberLayerPointers',oldNumbers);
end

[ismem memidx] = ismember(m.layer_map.minibatch_index,oldNumbers); m.layer_map.minibatch_index = memidx(ismem);
[ismem memidx] = ismember(m.layer_map.input,oldNumbers); m.layer_map.input = memidx(ismem);
[ismem memidx] = ismember(m.layer_map.output,oldNumbers); m.layer_map.output = memidx(ismem);

m = cnpkg4.ComputeNodeTopoOrder(m);

end

% Contains definitions that apply to an entire model.

%-----------------------------------------------------------------------------------------------------------------------

function f = CNSFields

% Pointers to mask and label blocks on global memory from which the mask
% and label patches are picked
% Points to the input block
f.inputblock = {'ma', 'mv', 'dnames', {'f' 'y' 'x' 'd'}, 'dims', {1 1 2 2}, 'dparts', {1 2 1 2}};
% Points to the label block
f.labelblock = {'ma', 'mv', 'dnames', {'f' 'y' 'x' 'd'}, 'dims', {1 1 2 2}, 'dparts', {1 2 1 2}};
% Points to the mask block
f.maskblock = {'ma', 'mv', 'dnames', {'f' 'y' 'x' 'd'}, 'dims', {1 1 2 2}, 'dparts', {1 2 1 2}};

f.offset = {'mp', 'mv', 'int', 'dflt', [0 0 0]};
f.globaleta = {'mp', 'dflt', 1.0};
f.binarythreshold = {'mp', 'dflt', 0.5};

end

%-----------------------------------------------------------------------------------------------------------------------

end
end
