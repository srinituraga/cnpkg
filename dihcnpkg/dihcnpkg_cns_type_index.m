function varargout = cnpkg_cns_type_index(method, varargin)

[varargout{1 : nargout}] = feval(['method_' method], varargin{:});

return;

%***********************************************************************************************************************

function p = method_props

p.dnames = {'d' 'i'};  % 'c' - co-ordinate axis along the y,x,d and 'i' = iterations
p.dims   = {2 1}; %Dimensionality mapping from matlab to GPU
p.dparts = {1 1};
p.kernel = false;

return;

%***********************************************************************************************************************

function d = method_fields

%d.val = {'cc', 'int', 'dflt', 0};
d.val = {'cv', 'int', 'dflt', 0};
d.additionalOffset = {'cv', 'int', 'dflt', 0};

return;