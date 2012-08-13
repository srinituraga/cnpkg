function varargout = demopkg_cns(method, varargin)

[varargout{1 : nargout}] = feval(['method_' method], varargin{:});

return;

%***********************************************************************************************************************

function p = method_props

p.methods = {'init'};

return;

%***********************************************************************************************************************

function d = method_fields

d = struct;

return;

%***********************************************************************************************************************

function m = method_init(m)

m = cns_setstepnos(m, 'field', 'pz');

for z = 1 : numel(m.layers)
    m = cns_type('initlayer', m, z);
end

m.independent = true;

return;