function varargout = cns_trans(mode, varargin)

% Internal CNS function.

%***********************************************************************************************************************

% Copyright (C) 2009 by Jim Mutch (www.jimmutch.com).
%
% This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public
% License as published by the Free Software Foundation; either version 3 of the License, or (at your option) any later
% version.
%
% This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
% warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License along with this program.  If not, see
% <http://www.gnu.org/licenses/>.

%***********************************************************************************************************************

switch mode
case 'parse'   , [varargout{1 : nargout}] = Parse   (varargin{:});
case 'undef'   , [varargout{1 : nargout}] = Undef   (varargin{:});
case 'create'  , [varargout{1 : nargout}] = Create  (varargin{:});
case 'recreate', [varargout{1 : nargout}] = Recreate(varargin{:});
case 'scalar'  , [varargout{1 : nargout}] = Scalar  (varargin{:});
case 'add'     , [varargout{1 : nargout}] = Add     (varargin{:});
case 'siz2p'   , [varargout{1 : nargout}] = Siz2P   (varargin{:});
case 'sizeis'  , [varargout{1 : nargout}] = SizeIs  (varargin{:});
case 'disp'    , [varargout{1 : nargout}] = Disp    (varargin{:});
case 'pack'    , [varargout{1 : nargout}] = Pack    (varargin{:});
case 'unpack'  , [varargout{1 : nargout}] = Unpack  (varargin{:});
case 'e2yx'    , [varargout{1 : nargout}] = E2YX    (varargin{:});
case 'yx2e'    , [varargout{1 : nargout}] = YX2E    (varargin{:});
case 'range'   , [varargout{1 : nargout}] = Range   (varargin{:});
otherwise      , error('invalid mode');
end

return;

%***********************************************************************************************************************

function d = Parse(d, u, shiftOK)

if ~isfield(u, 'dims') && isfield(u, 'dparts')
    error('dparts cannot be specified without dims');
end

if isfield(u, 'dims')
    dims = u.dims(:)';
    if ~iscell(dims) || isempty(dims) || ~all(cellfun(@isnumeric, dims))
        error('dims must be a cell array of numeric vectors');
    end
    for i = 1 : numel(dims)
        if isempty(dims{i}) || (size(dims{i}, 1) ~= 1)
            error('dims must contain scalars or nonempty row vectors');
        end
    end
    if ~any([dims{:}] == 1)
        error('dims must contain the number 1 at least once');
    end
    if any(([dims{:}] ~= 1) & ([dims{:}] ~= 2))
        error('dims may only contain the numbers 1 and 2');
    end
else
    dims = {1 2};
end

if isfield(u, 'dparts')
    dparts = u.dparts(:)';
    if ~iscell(dparts) || ~all(cellfun(@isnumeric, dparts))
        error('dparts must be a cell array of numeric vectors');
    end
    if numel(dparts) ~= numel(dims)
        error('dparts must have the same number of elements (%u) as dims', numel(dims));
    end
    for i = 1 : numel(dims)
        if ~isequal(size(dparts{i}), size(dims{i}))
            error('each element of dparts must be the same size as the corresponding element of dims');
        end
    end
    tdims  = [dims{:}  ];
    tparts = [dparts{:}];
    for i = 1 : max(tdims)
        if ~isequal(sort(tparts(tdims == i)), 1 : sum(tdims == i))
            error('parts for dimension %u must uniquely cover the range 1 to %u', i, sum(tdims == i));
        end
    end
else
    dparts = {};
    for i = 1 : numel(dims)
        for j = 1 : numel(dims{i})
            dparts{i}(1, j) = sum([dims{1 : i - 1}, dims{i}(1 : j - 1)] == dims{i}(j)) + 1;
        end
    end
end

if isfield(u, 'dnames')
    dnames = u.dnames(:)';
    if ~iscellstr(dnames)
        error('dnames must be a cell array of strings');
    end
    if numel(dnames) ~= numel(dims)
        error('dnames must have the same number of elements (%u) as dims', numel(dims));
    end
    if any(isstrprop([dnames{:}], 'digit'))
        error('dnames cannot contain digits');
    end
    tnames = dnames(~strcmp(dnames, ''));
    if numel(unique(tnames)) ~= numel(tnames)
        error('dnames must be unique');
    end
elseif isequal(dims, {1 2})
    dnames = {'y' 'x'};
else
    dnames = repmat({''}, 1, numel(dims));
end

if isfield(u, 'dmap')
    dmap = u.dmap(:)';
    if ~islogical(dmap) && ~isnumeric(dmap)
        error('dmap must be a logical or numeric vector');
    end
    dmap = double(dmap);
    if any(~ismember(dmap, [0 1 2])) || (~shiftOK && any(dmap == 2))
        error('invalid dmap value');
    end
    if numel(dmap) ~= numel(dims)
        error('dmap must have the same number of elements (%u) as dims', numel(dims));
    end
    if any(dmap & cellfun(@isempty, dnames))
        error('dmap cannot be true if dnames is null');
    end
    if any(dmap & ~cellfun(@isscalar, dims))
        error('mapped dimensions cannot have multiple parts');
    end
    if any(dmap == 2)
        if sum(dmap == 2) > 1, error('can only have one shiftable dimension'); end
        if (dims{dmap == 2} ~= 2) || (dparts{dmap == 2} ~= sum([dims{:}] == 2))
            error('shiftable dimension must be the last part of internal dimension 2');
        end
    end
else
    dmap = zeros(1, numel(dims));
end

d.dims   = dims;
d.dparts = dparts;
d.dnames = dnames;
d.dmap   = dmap;

return;

%***********************************************************************************************************************

function d = Undef(d)

d.dims   = {};
d.dparts = {};
d.dnames = {};
d.dmap   = [];

return;

%***********************************************************************************************************************

function t = Create(d, sizes, align)

if (nargin < 3) || isempty(align), align = 1; end

sizes = sizes(:)';
if ~iscell(sizes) || ~all(cellfun(@isnumeric, sizes))
    error('size must be a cell array of numeric vectors');
end
if numel(sizes) ~= numel(d.dims)
    error('size must have %u elements', numel(d.dims));
end
for i = 1 : numel(d.dims)
    if isequal(size(sizes{i}), size(d.dims{i}))
    elseif isscalar(sizes{i})
        sizes{i} = cns_splitdim(sizes{i}, numel(d.dims{i}));
    else
        error('size{%u} must be 1x%u or scalar', i, numel(d.dims{i}));
    end
end

counts = [sizes{:}];
if any(counts < 0) || any(mod(counts, 1) ~= 0)
    error('invalid size');
end

[rows, order] = sortrows([[d.dims{:}]', [d.dparts{:}]']);

t.split = ~cellfun(@isscalar, d.dims);
t.shift = (d.dmap == 2);

t.map = zeros(1, numel(order));
k = 0;
for i = 1 : numel(d.dims)
    for j = 1 : numel(d.dims{i})
        k = k + 1;
        t.map(k) = i;
    end
end

t.perm = order';

t.align = align;

t.npart = [];
for i = 1 : max(rows(:, 1))
    t.npart(i) = sum(rows(:, 1) == i);
end

t = Resize(t, counts);

return;

%***********************************************************************************************************************

function t = Recreate(d, siz2, varargin)

sizes = {};
pos = 1;

for i = 1 : numel(d.dims)
    for j = 1 : numel(d.dims{i})
        sizes{i}(j) = siz2(pos);
        pos = pos + 1;
    end
end

t = Create(d, sizes, varargin{:});

return;

%***********************************************************************************************************************

function t = Scalar(align)

if (nargin < 1) || isempty(align), align = 1; end

t.split = false(0);
t.shift = false(0);
t.map   = [];
t.perm  = [];
t.align = align;
t.npart = [];

t = Resize(t, []);

return;

%***********************************************************************************************************************

function t = Add(t, count)

n = numel(count);

t.split = [false(1, n), t.split];
t.shift = [false(1, n), t.shift];
t.map   = [1 : n, t.map + n];
t.perm  = [t.perm + n, n : -1 : 1];
t.npart = [t.npart, n];

t = Resize(t, [count, t.siz2]);

return;

%***********************************************************************************************************************

function s = Siz2P(t)

s = [t.siz2, t.siz3a(1), t.siz4b(1)];

return;

%***********************************************************************************************************************

function r = SizeIs(t, a)

d1 = numel(t.siz1);
d2 = numel(size(a));

if d1 == 0

    r = isscalar(a);

elseif (d1 == 1) && (d2 == 2)

    if all(size(a) ~= 1) && any(size(a) > 0)
        r = false;
    else
        r = (numel(a) == t.siz1);
    end
    
else
    
    r = true;
    for d = 1 : max(d1, d2)
        if d <= d1, s1 = t.siz1(d) ; else s1 = 1; end
        if d <= d2, s2 = size(a, d); else s2 = 1; end
        if s1 ~= s2
            r = false;
            break;
        end
    end
    
end

return;

%***********************************************************************************************************************

function s = Disp(t, mv)

if nargin < 2, mv = false; end

if isempty(t.siz1)
    s = '1';
elseif mv
    s = 'N';
else
    s = sprintf('%u', t.siz1(1));
end

if numel(t.siz1) > 1
    s = [s, sprintf('x%u', t.siz1(2 : end))];
end

return;

%***********************************************************************************************************************

function a = Pack(t, a, final, roll)

if roll > 0
    a = RollData(t.shift, t.siz1s, roll, true, a);
end

a = permute(reshape(a, t.siz2s), t.perms);

a(t.siz3(1) + 1 : t.siz3a(1), :) = 0;

a = reshape(a, t.siz4a);

if final
    a(t.siz4a(1) + 1 : t.siz4b(1), :) = 0;
    a = reshape(a, t.siz4b);
end

return;

%***********************************************************************************************************************

function a = Unpack(t, a, roll)

a = reshape(a(1 : t.siz4a(1), :), t.siz3a);

a = reshape(ipermute(reshape(a(1 : t.siz3(1), :), t.siz3), t.perms), t.siz1s);

if roll > 0
    a = RollData(t.shift, t.siz1s, roll, false, a);
end

return;

%***********************************************************************************************************************

function varargout = E2YX(t, e, roll)

if nargout < numel(t.nparts), error('not enough outputs'); end

[c2{1 : numel(t.siz2s)}] = ind2sub(t.siz2s, e);

if roll > 0
    [c2{:}] = RollCoords(t.shift(t.map), t.siz2s, roll, true, c2{:});
end

[varargout{1 : nargout}] = ind2sub(t.siz4a, sub2ind(t.siz3a, c2{t.perms}));

return;

%***********************************************************************************************************************

function e = YX2E(t, y, x, roll)

if numel(t.nparts) > 2, error('invalid for more than 2 internal dimensions'); end

[c2{t.perms}] = ind2sub(t.siz3a, sub2ind(t.siz4a, y, x));

if roll > 0
    [c2{:}] = RollCoords(t.shift(t.map), t.siz2s, roll, false, c2{:});
end

e = sub2ind(t.siz2s, c2{:});

return;

%***********************************************************************************************************************

function q = Range(t, shift, roll, varargin)

n = numel(varargin);
if (n > 0) && (n < numel(t.siz1))
    sizes = [t.siz1(1 : n - 1), prod(t.siz1(n : end))];
    split = [t.split(1 : n - 1), true];
    sflag = [t.shift(1 : n - 1), false];
    map   = min(t.map, n);
elseif n == numel(t.siz1)
    sizes = t.siz1;
    split = t.split;
    sflag = t.shift;
    map   = t.map;
else
    error('too many arguments');
end

lens   = zeros(1, n);
counts = t.siz2;
c1     = zeros(1, n);
c2     = zeros(1, n);
for i = 1 : n
    c = varargin{i};
    if ~isnumeric(c) || any(mod(c, 1) ~= 0), error('invalid index'); end
    switch numel(c)
    case 0   , a = 1   ; b = sizes(i);
    case 1   , a = c   ; b = c;
    case 2   , a = c(1); b = c(2);
    otherwise, error('too many arguments');
    end
    if sflag(i) && (a == 0) && (b == 0)
        a = sizes(i) - shift + 1;
        b = sizes(i);
    else
        if a < 0, a = sizes(i) + a + 1; end
        if b < 0, b = sizes(i) + b + 1; end
        if (a < 1) || (a > sizes(i) + 1), error('index out of range'); end
        if (b < a - 1) || (b > sizes(i)), error('index out of range'); end
    end
    lens(i) = b - a + 1;
    if split(i) && (lens(i) > 1) && (lens(i) < sizes(i))
        error('range not allowed for split dimension');
    end
    counts((map == i) & (counts > lens(i))) = lens(i); % maintains full size of split dimensions
    c1(i) = a;
    c2(i) = b;
end

q.t.split = split;
q.t.shift = sflag;
q.t.map   = map;
q.t.perm  = t.perm;
q.t.align = t.align;
q.t.npart = t.npart;

q.t = Resize(q.t, counts);

if any(lens == 0)
    q.yo = 0;
    q.yc = 0;
    q.xo = 0;
    q.xc = 0;
    q.fo = 0;
    q.fc = 0;
    return;
end

i = 0;
for d = 1 : numel(t.npart)
    partial = false;
    for p = 1 : t.npart(d)
        i = i + 1;
        j = map(t.perm(i));
        if partial
            if lens(j) > 1, error('requested values are not stored in contiguous memory'); end
        elseif lens(j) < sizes(j)
            partial = true;
        end
    end
end

if n == 0
    [y1, x1, f1] = deal(1, 1, 1);
    [y2, x2, f2] = deal(1, 1, 1);
else
    c1 = num2cell(c1);
    c2 = num2cell(c2);
    [y1, x1, f1] = E2YX(t, cns_iconv(sizes, [], c1{:}), roll);
    [y2, x2, f2] = E2YX(t, cns_iconv(sizes, [], c2{:}), roll);
end

q.yo = y1 - 1;
q.yc = y2 - y1 + 1;

if x1 <= x2
    q.xo = x1 - 1;
    q.xc = x2 - x1 + 1;
else
    q.xo = [x1 - 1, 0];
    q.xc = [t.siz4a(2) - x1 + 1, x2];
end

q.fo = f1 - 1;
q.fc = f2 - f1 + 1;

return;

%***********************************************************************************************************************

function t = Resize(t, counts, align)

if (numel(counts) < numel(t.map)) || any(counts(numel(t.map) + 1 : end) ~= 1), error('invalid size'); end
counts = counts(1 : numel(t.map));

if (nargin >= 3) && ~isempty(align)
    t.align = align;
end

t.perms = t.perm;
t.perms(end + 1 : 2) = numel(t.perms) + 1 : 2;

t.nparts = t.npart;
t.nparts(end + 1 : 2) = zeros(1, 2 - numel(t.nparts));

t.siz1 = [];
for i = 1 : numel(t.shift)
    t.siz1(i) = prod(counts(t.map == i));
end
t.siz1s = t.siz1;
t.siz1s(end + 1 : 2) = ones(1, 2 - numel(t.siz1s));

t.siz2 = counts;
t.siz2s = t.siz2;
t.siz2s(end + 1 : 2) = ones(1, 2 - numel(t.siz2s));

t.siz3 = t.siz2s(t.perms);

% Align dimension 1 if most positions will still be valid.  If we don't, we still have to pad the final result (see
% siz4b below).  NOTE: disabled because tests so far seem to indicate that it's always better to pack the threads.
t.siz3a = t.siz3;
% if t.align ~= 1
%     yCount0 = t.siz3(1);
%     ySize0 = ceil(yCount0 / t.align) * t.align;
%     if yCount0 / ySize0 >= 0.75, t.siz3a(1) = ySize0; end
% end

t.siz4a = ones(1, numel(t.nparts));
k = 0;
for i = 1 : numel(t.nparts)
    t.siz4a(i) = prod(t.siz3a(k + 1 : k + t.nparts(i)));
    k = k + t.nparts(i);
end

t.siz4b = t.siz4a;
if t.align ~= 1
    t.siz4b(1) = ceil(t.siz4a(1) / t.align) * t.align;
end

return;

%***********************************************************************************************************************

function a = RollData(flags, sizes, roll, pack, a)

coords = cell(1, numel(sizes));

for i = 1 : numel(sizes)
    if (i > numel(flags)) || ~flags(i)
        coords{i} = ':';
    elseif pack
        coords{i} = [roll + 1 : sizes(i), 1 : roll];
    else
        coords{i} = zeros(1, sizes(i));
        coords{i}([roll + 1 : sizes(i), 1 : roll]) = 1 : sizes(i);
    end
end

a = a(coords{:});

return;

%***********************************************************************************************************************

function varargout = RollCoords(flags, sizes, roll, pack, varargin)

for i = 1 : numel(sizes)
    c = varargin{i};
    if (i > numel(flags)) || ~flags(i)
    elseif pack
        c = c - roll;
        if c < 1, c = c + sizes(i); end
    else
        c = c + roll;
        if c > sizes(i), c = c - sizes(i); end
    end
    varargout{i} = c;
end

return;