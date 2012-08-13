function varargout = cnpkg_cns_type_mattcomputed(method, varargin)

[varargout{1 : nargout}] = feval(['method_' method], varargin{:});

return;

%***********************************************************************************************************************

function p = method_props

p.super    = 'layer';
p.abstract = true;

return;

%***********************************************************************************************************************

function d = method_fields

d.zp = {'lz', 'type', 'layer'};
d.zw = {'lz', 'type', 'weight'};
d.zb = {'lz', 'type', 'bias'};

d.sens = {'cv', 'dflt', 0};
d.preact = {'cv', 'dflt', 0};

return;