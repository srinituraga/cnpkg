function varargout = cnpkg_cns_type_label(method, varargin)

[varargout{1 : nargout}] = feval(['method_' method], varargin{:});

return;

%***********************************************************************************************************************

function p = method_props

p.super  = 'label';
p.blockYSize = 16;
p.blockXSize = 4;

return;

%***********************************************************************************************************************

function d = method_fields

d = struct;

% nhood1 & nhood2
d.nhood1 = {'la', 'cache', 'int', 'dnames', {'i' 'd'}, 'dims', {1 2}, 'dparts', {1 1}};
d.nhood2 = {'la', 'cache', 'int', 'dnames', {'i' 'd'}, 'dims', {1 2}, 'dparts', {1 1}};

return;
