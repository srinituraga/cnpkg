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

d.offset = {'mp', 'mv', 'int', 'dflt', [0 0 0]};
d.globaleta = {'mp', 'dflt', 1.0};
d.binarythreshold = {'mp', 'dflt', 0.5};

return;
