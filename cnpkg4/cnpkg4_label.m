classdef cnpkg4_label < cnpkg4_node
methods (Static)

%***********************************************************************************************************************

function p = CNSProps

p = struct;

end

%***********************************************************************************************************************

function f = CNSFields

% pointer to minibatch indices
f.zin = {'lz', 'type', 'index'};
% mask variable (same size as the labels)
f.mask = {'cv', 'cache', 'dflt', 0};

end

%***********************************************************************************************************************

function dep = GetDependencies(m,node)
    dep = [];
end

%***********************************************************************************************************************

function m = RenumberLayerPointers(m,node,oldNumbers)
    m = cns_super(m,node,oldNumbers);
    [ismem memidx] = ismember(m.layers{node}.zin,oldNumbers); m.layers{node}.zin = memidx(ismem);
end

%***********************************************************************************************************************
end
end
