function varargout = cnpkg_cns_type_layer(method, varargin)

[varargout{1 : nargout}] = feval(['method_' method], varargin{:});

return;

%***********************************************************************************************************************

function p = method_props

p.abstract = true;

p.dims   = {1 1 2 2 1};
p.dparts = {2 1 1 2 3};
p.dnames = {'f' 'y' 'x' 'd' 'n'};
p.dmap   = [false true true true false];

return;

%***********************************************************************************************************************

function d = method_fields

d.val = {'cv', 'dflt', 0};

return;
