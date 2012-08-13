function varargout = cnpkg_cns_type_base(method, varargin)

[varargout{1 : nargout}] = feval(['method_' method], varargin{:});

return;

%***********************************************************************************************************************

function p = method_props

p.abstract = true;

return;

%***********************************************************************************************************************

function d = method_fields

d = struct;

return;