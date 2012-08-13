function varargout = cnpkg_cns_type_outputSqSq(method, varargin)

[varargout{1 : nargout}] = feval(['method_' method], varargin{:});

return;

%***********************************************************************************************************************

function p = method_props

p.super = 'computed';

p.blockYSize = 16;
p.blockXSize = 4;

return;

%***********************************************************************************************************************

function d = method_fields

d.zl  = {'lz', 'type', 'label'};
d.margin  = {'lp', 'dflt', 0.5};

% Patches that store the actual label and mask variables
d.loss = {'cv', 'cache', 'dflt', 0}; % the loss per pixel
d.classerr = {'cv', 'cache', 'dflt', 0}; % the loss per pixel


return;
