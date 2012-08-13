classdef cnpkg4_computed < cnpkg4_node
methods (Static)

%***********************************************************************************************************************

function p = CNSProps

p = struct;

end

%***********************************************************************************************************************

function f = CNSFields

f.zp = {'lz', 'type', 'node', 'mv'};
f.zn = {'lz', 'type', 'node', 'mv'};
f.zw = {'lz', 'type', 'weight', 'mv'};
f.zb = {'lz', 'type', 'bias'};

end

%***********************************************************************************************************************

function dep = GetDependencies(m,node)
    dep = m.layers{node}.zp;
end

%***********************************************************************************************************************

function m = RenumberLayerPointers(m,node,oldNumbers)
    m = cns_super(m,node,oldNumbers);
    [ismem memidx] = ismember(m.layers{node}.zp,oldNumbers); m.layers{node}.zp = memidx(ismem);
    [ismem memidx] = ismember(m.layers{node}.zn,oldNumbers); m.layers{node}.zn = memidx(ismem);
    [ismem memidx] = ismember(m.layers{node}.zw,oldNumbers); m.layers{node}.zw = memidx(ismem);
    [ismem memidx] = ismember(m.layers{node}.zb,oldNumbers); m.layers{node}.zb = memidx(ismem);
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
            st = m.layers{nextNode}.([dims(iDim) '_start'])-filtOffset*m.layers{node}.([dims(iDim) '_space']);
            % how big should this layer be?
            nextsz = (m.layers{nextNode}.size{iDim+1}-1)*m.layers{nextNode}.([dims(iDim) '_space'])+1;
            sz = ceil(nextsz/m.layers{node}.([dims(iDim) '_space']))+2*filtOffset;

            if ~isfield(m.layers{node},[dims(iDim) '_start']),
                m.layers{node}.([dims(iDim) '_start']) = st;
                m.layers{node}.size{iDim+1} = sz;
            else,
                m.layers{node}.([dims(iDim) '_start']) = min(m.layers{node}.([dims(iDim) '_start']), st);
                m.layers{node}.size{iDim+1} = max(m.layers{node}.size{iDim+1}, sz);
            end

        % IS PROBABLY BUGGY!! AND DEFINITELY NOT TESTED YET!! SHOULD ALSO REWRITE KERNELS TO HANDLE FULL CONVS
        case 'full'
            % where should this node start?
            st = m.layers{nextNode}.([dims(iDim) '_start'])+filtOffset*m.layers{node}.([dims(iDim) '_space']);
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
if ~isempty(m.layers{node}.zn),
    m.layers{node}.size{5} = m.layers{nextNode}.size{5};
end
end

%***********************************************************************************************************************

function m = MapDimFwd(m,node)

dims = ['y', 'x', 'd'];

for iDim = 1:length(dims),
    for iPrevNode = 1:length(m.layers{node}.zp),
        prevNode = m.layers{node}.zp(iPrevNode);
        prevWeight = m.layers{node}.zw(iPrevNode);

        if ~isfield(m.layers{prevNode},'size'),
            error('found un-initialized prev node! maybe not in toposort order?')
        end

        % figure out field of view of the filter
        filtOffset = floor(m.layers{prevWeight}.size{1+iDim}/2) ...
                            * m.layers{prevWeight}.([dims(iDim) '_space']);

        % what type of convolution is it?
        switch m.layers{prevWeight}.convType,
        case 'full'
            % where should this node start?
            st = m.layers{prevNode}.([dims(iDim) '_start'])-filtOffset;
            m.layers{node}.([dims(iDim) '_start']) = min(m.layers{node}.([dims(iDim) '_start']), st);
            % how big should this layer be?
            prevsz = (m.layers{prevNode}.size{iDim+1}-1)*m.layers{prevNode}.([dims(iDim) '_space'])+1;
            sz = floor((prevsz+2*filtOffset)/m.layers{node}.([dims(iDim) '_space']));
            m.layers{node}.size{iDim+1} = max(m.layers{node}.size{iDim+1}, sz);
        case 'valid'
            % where should this node start?
            st = m.layers{prevNode}.([dims(iDim) '_start'])+filtOffset;
            m.layers{node}.([dims(iDim) '_start']) = max(m.layers{node}.([dims(iDim) '_start']), st);
            % how big should this layer be?
            prevsz = (m.layers{prevNode}.size{iDim+1}-1)*m.layers{prevNode}.([dims(iDim) '_space'])+1;
            sz = floor((prevsz-2*filtOffset)/m.layers{node}.([dims(iDim) '_space']));
            m.layers{node}.size{iDim+1} = min(m.layers{node}.size{iDim+1}, sz);
        end
    end
end
end

%***********************************************************************************************************************
end
end
