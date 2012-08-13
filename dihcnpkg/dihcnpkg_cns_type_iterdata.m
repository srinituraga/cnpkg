function varargout = dihcnpkg_cns_type_iterdata(method, varargin)

[varargout{1 : nargout}] = feval(['method_' method], varargin{:});

return;

%***********************************************************************************************************************

function p = method_props

p.dnames = {'d' 'i'};  % 'd' - diff data values and 'i' = iterations
p.dims   = {2 1}; %Dimensionality mapping from matlab to GPU
p.dparts = {1 1};
p.blockYSize = 32;
p.blockXSize = 2;
% p.kernel = false;

return;

%***********************************************************************************************************************

function d = method_fields

%d.val = {'cc', 'int', 'dflt', 0};
d.zout = {'lz', 'type', 'output'};
d.err = {'cv', 'dflt', 0};

return;