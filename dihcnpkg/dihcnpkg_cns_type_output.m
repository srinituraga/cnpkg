function varargout = cnpkg_cns_type_output(method, varargin)

[varargout{1 : nargout}] = feval(['method_' method], varargin{:});

return;

%***********************************************************************************************************************

function p = method_props

p.super = 'computed';

p.blockYSize = 16;
%p.blockXSize = 16;
p.blockXSize = 8;

return;

%***********************************************************************************************************************

function d = method_fields

d.zin = {'lz', 'type', 'index'};
%d.zim = {'lp', 'type', 'image'};
% d.zmask = {'lp', 'type', 'image'};

% d.labelm = {'cv', 'dflt', 0};
d.label = {'cv', 'dflt', 0};
% d.blaha = {'cv', 'dflt', 0};
% d.blahb = {'cv', 'dflt', 0};
% d.blahc = {'cv', 'dflt', 0};
d.labelblock = {'la', 'dnames', {'y' 'x' 'd'}, 'dims', {1 2 1}, 'dparts', {1 1 2}};
% d.labelmaskblock = {'la', 'dnames', {'y' 'x' 'd'}, 'dims', {1 2 1}, 'dparts', {1 1 2}};

return;
