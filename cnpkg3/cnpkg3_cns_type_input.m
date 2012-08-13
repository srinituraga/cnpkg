function varargout = cnpkg_cns_type_input(method, varargin)

[varargout{1 : nargout}] = feval(['method_' method], varargin{:});

return;

%***********************************************************************************************************************

function p = method_props

p.super  = 'layer';
p.blockYSize = 16;
p.blockXSize = 4;

return;

%***********************************************************************************************************************

function d = method_fields

d = struct;
d.zin = {'lz', 'type', 'index'};

return;
