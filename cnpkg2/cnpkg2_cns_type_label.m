function varargout = cnpkg_cns_type_label(method, varargin)

[varargout{1 : nargout}] = feval(['method_' method], varargin{:});

return;

%***********************************************************************************************************************

function p = method_props

p.super  = 'layer';
p.blockYSize = 16;
p.blockXSize = 4;

return;

%***********************************************************************************************************************

function d = method_fields

d = struct;
% pointer to minibatch indices
d.zin = {'lz', 'type', 'index'};
% mask variable (same size as the labels)
d.mask = {'cv', 'cache', 'dflt', 0};

% Pointers to mask and label blocks on global memory from which the mask
% and label patches are picked
% Points to the label block
d.labelblock = {'la', 'mv', 'dnames', {'f' 'y' 'x' 'd'}, 'dims', {1 1 2 2}, 'dparts', {1 2 1 2}};
% Points to the mask block
d.maskblock = {'la', 'mv', 'dnames', {'f' 'y' 'x' 'd'}, 'dims', {1 1 2 2}, 'dparts', {1 2 1 2}};

return;
