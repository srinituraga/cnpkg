function varargout = cnpkg_cns_type_image(method, varargin)

[varargout{1 : nargout}] = feval(['method_' method], varargin{:});

return;

%***********************************************************************************************************************

function p = method_props

p.super  = 'layer';
p.kernel = false;

%p.dims   = {1 1 2};
%p.dparts = {2 1 1};
%p.dnames = {'y' 'x' 'd'};
%p.dmap   = [true true true];

return;

%***********************************************************************************************************************

function d = method_fields

d = struct;


%d.image = {'cv', 'dflt', 0};

return;