function varargout = demopkg_cns_type_base(method, varargin)

[varargout{1 : nargout}] = feval(['method_' method], varargin{:});

return;

%***********************************************************************************************************************

function p = method_props

p.abstract = true;
p.methods  = {'initlayer'};

p.dims   = {1 1 2};
p.dparts = {2 1 1};
p.dnames = {'f' 'y' 'x'};
p.dmap   = [false true true];

return;

%***********************************************************************************************************************

function d = method_fields

d.pz  = {'lz', 'type', 'base'};
d.val = {'cv', 'cache', 'dflt', 0};

return;

%***********************************************************************************************************************

function m = method_initlayer(m, z)

return;