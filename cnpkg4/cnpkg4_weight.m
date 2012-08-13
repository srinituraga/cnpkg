classdef cnpkg4_weight < cnpkg4_base
methods (Static)

%***********************************************************************************************************************

function p = CNSProps

p.dnames = {'f' 'y' 'x' 'd' 'nf'};
p.dims   = {1 1 2 2 1};
p.dparts = {2 1 1 2 3};
p.kernels = {'constrain'};
end

%***********************************************************************************************************************

function f = CNSFields

f.zp = {'lz', 'type', 'node'};
f.zs = {'lz', 'type', 'sens', 'mv'};

f.y_space = {'lp','dflt',1};
f.x_space = {'lp','dflt',1};
f.d_space = {'lp','dflt',1};
f.y_blk = {'lp','dflt',1};
f.x_blk = {'lp','dflt',1};
f.d_blk = {'lp','dflt',1};

f.val = {'cv', 'cache'};
f.dval = {'cv', 'cache'};

f.eta = {'lp'};
f.gradonly = {'lp','dflt',0};

end

%***********************************************************************************************************************

function dep = GetDependencies(m,node)
    if (isfield(m,'globaleta') && (m.globaleta == 0)) || (m.layers{node}.eta == 0),
        dep = [];
    else,
        dep = [m.layers{node}.zp m.layers{node}.zs];
    end
end

%***********************************************************************************************************************

function m = RenumberLayerPointers(m,node,oldNumbers)
    m = cns_super(m,node,oldNumbers);
    [ismem memidx] = ismember(m.layers{node}.zp,oldNumbers); m.layers{node}.zp = memidx(ismem);
    [ismem memidx] = ismember(m.layers{node}.zs,oldNumbers); m.layers{node}.zs = memidx(ismem);
    % if no sensitivity exists, then assume testing mode
    % so set the eta to 0
    if isempty(m.layers{node}.zs)
        m.layers{node}.eta = 0; % 
    end
end

%***********************************************************************************************************************
end
end
