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

% Pointers to mask and label blocks on global memory from which the mask
% and label patches are picked
% Points to the input block
d.inputblock = {'ma', 'mv', 'dnames', {'f' 'y' 'x' 'd'}, 'dims', {1 1 2 2}, 'dparts', {1 2 1 2}};
% Points to the label block
d.labelblock = {'ma', 'mv', 'dnames', {'f' 'y' 'x' 'd'}, 'dims', {1 1 2 2}, 'dparts', {1 2 1 2}};
% Points to the mask block
d.maskblock = {'ma', 'mv', 'dnames', {'f' 'y' 'x' 'd'}, 'dims', {1 1 2 2}, 'dparts', {1 2 1 2}};

d.offset = {'mp', 'mv', 'int', 'dflt', [0 0 0]};
d.globaleta = {'mp', 'dflt', 1.0};
d.binarythreshold = {'mp', 'dflt', 0.5};

return;
