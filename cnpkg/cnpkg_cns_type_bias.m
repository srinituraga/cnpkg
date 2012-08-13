function varargout = cnpkg_cns_type_bias(method, varargin)

[varargout{1 : nargout}] = feval(['method_' method], varargin{:});

return;

%***********************************************************************************************************************

function p = method_props

p.dims   = {1 2};
p.dparts = {1 1};
p.dnames = {'nf' ''};

p.blockYSize = 16;
p.blockXSize = 16;

return;

%***********************************************************************************************************************

function d = method_fields

d.zn = {'lz', 'type', 'computed'};

d.eta = {'lp'};

d.val = {'cv', 'dflt', 0.5};
d.dval = {'cv', 'dflt', 0.5};

return;
