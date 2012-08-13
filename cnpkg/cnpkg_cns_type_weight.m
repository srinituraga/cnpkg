function varargout = cnpkg_cns_type_weight(method, varargin)

[varargout{1 : nargout}] = feval(['method_' method], varargin{:});

return;

%***********************************************************************************************************************

function p = method_props

p.dims   = {1 1 2 2 1};
p.dparts = {2 1 1 2 3};
p.dnames = {'f' 'y' 'x' 'd' 'nf'};

p.blockYSize = 16;
p.blockXSize = 16;

return;

%***********************************************************************************************************************

function d = method_fields

d.zp = {'lz', 'type', 'layer'};
d.zn = {'lz', 'type', 'computed'};

d.eta = {'lp'};

d.val = {'cv', 'dflt', 0.5};
d.dval = {'cv', 'dflt', 0.5};

return;
