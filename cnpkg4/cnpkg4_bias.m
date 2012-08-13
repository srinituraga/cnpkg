classdef cnpkg4_bias < cnpkg4_base
methods (Static)

%***********************************************************************************************************************

function p = CNSProps

p.dims   = {1 2};
p.dparts = {1 1};
p.dnames = {'nf' ''};

end

%***********************************************************************************************************************

function f = CNSFields

f.zs = {'lz', 'type', 'sens', 'mv'};

f.eta = {'lp'};
f.gradonly = {'lp','dflt',0};

f.val = {'cv', 'cache', 'dflt', 0.5};
f.dval = {'cv', 'cache', 'dflt', 0.5};

end

%***********************************************************************************************************************

function dep = GetDependencies(m,node)
    if (isfield(m,'globaleta') && (m.globaleta == 0)) || (m.layers{node}.eta == 0),
        dep = [];
    else,
        dep = m.layers{node}.zs;
    end
end

%***********************************************************************************************************************

function m = RenumberLayerPointers(m,node,oldNumbers)
    m = cns_super(m,node,oldNumbers);
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
