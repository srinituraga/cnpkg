function varargout = cnpkg_cns_type_matthidden(method, varargin)

[varargout{1 : nargout}] = feval(['method_' method], varargin{:});

return;

%***********************************************************************************************************************

function p = method_props

p.super = 'computed';

p.blockYSize = 16;
%p.blockXSize = 16;
p.blockXSize = 4;

return;

%***********************************************************************************************************************

function d = method_fields

d.znw = {'lz', 'type', 'weight'};
d.zn  = {'lz', 'type', 'computed'};

d.presens = {'cv', 'dflt', 0};

return;
