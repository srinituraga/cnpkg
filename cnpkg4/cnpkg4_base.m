classdef cnpkg4_base < cns_base
methods (Static)

% Base type for all layers (including weight and bias layers).

%-----------------------------------------------------------------------------------------------------------------------

function p = CNSProps

% Because the base type is abstract, we don't have to define its dimensionality.  That allows subtypes to have
% different dimensionalities.

p.abstract = true;

end

%***********************************************************************************************************************

function m = RenumberLayerPointers(m,node,oldNumbers)
    if isfield(m.layers{node},'z'),
        [ismem memidx] = ismember(m.layers{node}.z,oldNumbers); m.layers{node}.z = memidx(ismem);
    end
end

%-----------------------------------------------------------------------------------------------------------------------

end
end
