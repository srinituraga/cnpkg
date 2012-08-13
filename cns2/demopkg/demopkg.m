classdef demopkg < cns_package
methods (Static)

%-----------------------------------------------------------------------------------------------------------------------

function m = CNSInit(m)

m = cns_setstepnos(m, 'field', 'pz');

for z = 1 : numel(m.layers)
    m = cns_call(m, z, 'InitLayer');
end

m.independent = true;

end

%-----------------------------------------------------------------------------------------------------------------------

end
end