classdef cnpkg4_error < cnpkg4_sens
methods (Static)

%***********************************************************************************************************************

function p = CNSProps

p = struct;
p.kernels = {'SqSq'};

end

%***********************************************************************************************************************

function f = CNSFields

f.zl  = {'lz', 'type', 'label'};
f.param  = {'lp', 'dflt', 0.5};

% Patches that store the actual label and mask variables
f.loss = {'cv', 'cache', 'dflt', 0}; % the loss per pixel
f.classerr = {'cv', 'cache', 'dflt', 0}; % the loss per pixel


end

%***********************************************************************************************************************

function dep = GetDependencies(m,node)
    dep = [m.layers{node}.zme m.layers{node}.zl];
end

%***********************************************************************************************************************

function m = RenumberLayerPointers(m,node,oldNumbers)
    m = cns_super(m,node,oldNumbers);
    [ismem memidx] = ismember(m.layers{node}.zl,oldNumbers); m.layers{node}.zl = memidx(ismem);
end

%***********************************************************************************************************************

function m = MapDimBkwd(m,node)
m.layers{node}.val = 0;
m.layers{node}.loss = 0;
m.layers{node}.classerr = 0;

m = cns_super(m,node);
for dim = 2:4,
    m = cns_mapdim(m, m.layers{node}.zl, dim, 'copy', node);
end
m.layers{m.layers{node}.zl}.val = 0;
m.layers{m.layers{node}.zl}.mask = 0;
m.layers{m.layers{node}.zl}.size{5} = m.layers{node}.size{5};

end

%***********************************************************************************************************************

function m = MapDimFwd(m,node)
m.layers{node}.val = 0;
m.layers{node}.loss = 0;
m.layers{node}.classerr = 0;

m = cns_super(m,node);
for dim = 2:4,
    m = cns_mapdim(m, m.layers{node}.zl, dim, 'copy', node);
end
m.layers{m.layers{node}.zl}.size{5} = m.layers{node}.size{5};

end


%***********************************************************************************************************************
end
end
