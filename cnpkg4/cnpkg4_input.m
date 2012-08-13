classdef cnpkg4_input < cnpkg4_node
methods (Static)

%***********************************************************
function m = SetInputSize(m,inputSize,nMiniBatch)
% setup the size of the input node
for k=1:3,
    m.layers{l}.size{1+k} = inputSize(k);
end
m.layers{l}.size{5} = nMiniBatch;
end


%***********************************************************************************************************************

function p = CNSProps

p = struct;

end

%***********************************************************************************************************************

function f = CNSFields

f.zin = {'lz', 'type', 'index', 'mv'};
f.zn = {'lz', 'type', 'node', 'mv'};

end

%***********************************************************************************************************************

function dep = GetDependencies(m,node)
    %dep = m.layers{node}.zin;
    dep = [];
end

%***********************************************************************************************************************

function m = RenumberLayerPointers(m,node,oldNumbers)
    m = cns_super(m,node,oldNumbers);
    [ismem memidx] = ismember(m.layers{node}.zin,oldNumbers); m.layers{node}.zin = memidx(ismem);
    [ismem memidx] = ismember(m.layers{node}.zn,oldNumbers); m.layers{node}.zn = memidx(ismem);
end

%***********************************************************************************************************************

function m = MapDimBkwd(m,node)

dims = ['y', 'x', 'd'];
m.layers{node}.val = 0;

for iDim = 1:length(dims),
    for iNextNode = 1:length(m.layers{node}.zn),
        nextNode = m.layers{node}.zn(iNextNode);
        nextWeight = m.layers{nextNode}.zw(ismember(m.layers{nextNode}.zp,node));

        if ~isfield(m.layers{nextNode},'size'),
            error('found un-initialized next node! maybe not in toposort order?')
        end

        % figure out field of view of the filter
        filtOffset = floor(m.layers{nextWeight}.size{1+iDim}/2) ...
                            * m.layers{nextWeight}.([dims(iDim) '_space']);

        % what type of convolution is it?
        switch m.layers{nextWeight}.convType,
        case 'valid'
            % where should this node start?
            st = m.layers{nextNode}.([dims(iDim) '_start'])-filtOffset;
            % how big should this layer be?
            nextsz = (m.layers{nextNode}.size{iDim+1}-1)*m.layers{nextNode}.([dims(iDim) '_space'])+1;
            sz = floor((nextsz+2*filtOffset)/m.layers{node}.([dims(iDim) '_space']));

            if ~isfield(m.layers{node},[dims(iDim) '_start']),
                m.layers{node}.([dims(iDim) '_start']) = st;
                m.layers{node}.size{iDim+1} = sz;
            else,
                m.layers{node}.([dims(iDim) '_start']) = min(m.layers{node}.([dims(iDim) '_start']), st);
                m.layers{node}.size{iDim+1} = max(m.layers{node}.size{iDim+1}, sz);
            end

        case 'full'
            % where should this node start?
            st = m.layers{nextNode}.([dims(iDim) '_start'])+filtOffset;
            % how big should this layer be?
            nextsz = (m.layers{nextNode}.size{iDim+1}-1)*m.layers{nextNode}.([dims(iDim) '_space'])+1;
            sz = floor((nextsz-2*filtOffset)/m.layers{node}.([dims(iDim) '_space']));

            if ~isfield(m.layers{node},[dims(iDim) '_start']),
                m.layers{node}.([dims(iDim) '_start']) = st;
                m.layers{node}.size{iDim+1} = sz;
            else,
                m.layers{node}.([dims(iDim) '_start']) = max(m.layers{node}.([dims(iDim) '_start']), st);
                m.layers{node}.size{iDim+1} = min(m.layers{node}.size{iDim+1}, sz);
            end
        end
    end
end
m.layers{node}.size{5} = m.layers{nextNode}.size{5};
if ~isempty(m.layers{node}.zin),
    m.layers{m.layers{node}.zin}.size{3} = m.layers{node}.size{5};
end
end


%***********************************************************************************************************************
end
end
