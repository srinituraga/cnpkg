classdef cnpkg4_sens < cnpkg4_node
methods (Static)

%***********************************************************************************************************************

function p = CNSProps

p = struct;

end

%***********************************************************************************************************************

function f = CNSFields

% f.sens = {'cv', 'cache', 'dflt', 0};
f.zme = {'lz', 'type', 'computed'};

f.znw = {'lz', 'type', 'weight', 'mv'};
f.zn  = {'lz', 'type', 'sens', 'mv'};


end

%***********************************************************************************************************************

function dep = GetDependencies(m,node)
    dep = [m.layers{node}.zme m.layers{node}.zn];
end

%***********************************************************************************************************************

function m = RenumberLayerPointers(m,node,oldNumbers)
    m = cns_super(m,node,oldNumbers);
    [ismem memidx] = ismember(m.layers{node}.zme,oldNumbers); m.layers{node}.zme = memidx(ismem);
    [ismem memidx] = ismember(m.layers{node}.znw,oldNumbers); m.layers{node}.znw = memidx(ismem);
    [ismem memidx] = ismember(m.layers{node}.zn,oldNumbers); m.layers{node}.zn = memidx(ismem);
end

%***********************************************************************************************************************

function m = MapDimFwd(m,node)
m.layers{node}.val = 0;

for dim = 2:4,
    m = cns_mapdim(m, node, dim, 'copy', m.layers{node}.zme);
end
m.layers{node}.size{5} = m.layers{m.layers{node}.zme}.size{5};

end

%***********************************************************************************************************************

function m = MapDimBkwd(m,node)
m.layers{node}.val = 0;

m.layers{node}.val = 0;
for dim = 2:4,
    m = cns_mapdim(m, node, dim, 'copy', m.layers{node}.zme);
end
m.layers{node}.size{5} = m.layers{m.layers{node}.zme}.size{5};

end

%***********************************************************************************************************************
end
end
