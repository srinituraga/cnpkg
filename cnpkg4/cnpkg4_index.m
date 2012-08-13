classdef cnpkg4_index < cnpkg4_base
methods (Static)

%***********************************************************************************************************************

function p = CNSProps

p.dims   = {2 1 2}; %Dimensionality mapping from matlab to GPU
p.dparts   = {1 1 2}; %Dimensionality mapping from matlab to GPU
p.dnames = {'d' 'i' 'n'};  % 'c' - co-ordinate axis along the y,x,d and 'i' = iterations
p.kernel = false;

end

%***********************************************************************************************************************

function f = CNSFields

f.val = {'cv', 'int', 'dflt', 0};

end

%***********************************************************************************************************************

function dep = GetDependencies(m,node)
    dep = [];
end

%***********************************************************************************************************************
end
end
