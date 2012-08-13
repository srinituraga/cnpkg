function varargout = cnpkg_cns_type_image(method, varargin)

[varargout{1 : nargout}] = feval(['method_' method], varargin{:});

return;

%***********************************************************************************************************************

function p = method_props

p.super  = 'layer';
p.kernel = false;

return;

%***********************************************************************************************************************

function d = method_fields

d = struct;

return;