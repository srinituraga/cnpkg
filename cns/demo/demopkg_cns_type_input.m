function varargout = demopkg_cns_type_input(method, varargin)

[varargout{1 : nargout}] = feval(['method_' method], varargin{:});

return;

%***********************************************************************************************************************

function p = method_props

p.kernel = false;

return;

%***********************************************************************************************************************

function d = method_fields

d = struct;

return;