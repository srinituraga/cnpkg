function varargout = cnpkg_cns(method, varargin)

[varargout{1 : nargout}] = feval(['method_' method], varargin{:});

return;

%***********************************************************************************************************************

function p = method_props

p = struct;

return;

%***********************************************************************************************************************

function d = method_fields

d = struct;

return;