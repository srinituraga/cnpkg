function varargout = demopkg_cns_type_scale(method, varargin)

[varargout{1 : nargout}] = feval(['method_' method], varargin{:});

return;

%***********************************************************************************************************************

function p = method_props

p.blockYSize = 16;
p.blockXSize = 8;

return;

%***********************************************************************************************************************

function d = method_fields

d = struct;

return;