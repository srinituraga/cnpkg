function cns_transgen(mode, d, printf, platform, name, varargin)

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

p.idims  = [];
p.iparts = [];
p.edims  = [];
p.pow    = false(0);
p.shift  = false(0);
for ed = 1 : numel(d.dims)
    for ep = 1 : numel(d.dims{ed})
        p.idims (end + 1) = d.dims  {ed}(ep);
        p.iparts(end + 1) = d.dparts{ed}(ep);
        p.edims (end + 1) = ed;
        p.pow   (end + 1) = (ep < numel(d.dims{ed}));
        p.shift (end + 1) = (d.dmap(ed) == 2);
    end
end

p.ned = max(p.edims);

% Index of the first 'y' coordinate.  This gets padded to siz3a(1).
p.ac = find((p.idims == 1) & (p.iparts == 1));

p.a3 = numel(p.idims) + 1; % position of siz3a(1)
p.a4 = numel(p.idims) + 2; % position of siz4b(1)
p.rc = numel(p.idims) + 3; % position of roll amount

switch platform
case 'cuda', p.attr = '__device__';
case 'cpu' , p.attr = 'inline';
end

p.ivars = 'yx';

p.platform = platform;
p.name     = name;

switch mode
case 'handle', HandleFunc (p, printf, varargin{:});
case 'size'  , SizeFuncs  (p, printf, varargin{:});
case 'coord' , CoordFuncs (p, printf, varargin{:});
case 'init'  , InitFunc   (p, printf, varargin{:});
case 'lookup', LookupFuncs(p, printf, varargin{:});
case 'select', SelectFunc (p, printf, varargin{:});
case 'write' , WriteFunc  (p, printf, varargin{:});
end

return;

%***********************************************************************************************************************

function HandleFunc(p, printf, type, head)

% type = (a)rray, (t)exture, (c)ommon, (n)euron, or public (w)
% head = declaration or definition?

if head

    printf('struct %s_h {\n', p.name);
    printf('const unsigned short *siz2p;\n');
    if ismember(type, {'c'})
        printf('unsigned short meta[2];\n');
    end
    printf('};\n');

else

    printf('%s %s_h %s_mh(', p.attr, p.name, p.name);
    if ismember(type, {'c'})
        printf('const unsigned short *meta, ');
    end
    printf('const unsigned short *siz2p) {\n');
    printf('%s_h h;\n', p.name);
    printf('h.siz2p = siz2p;\n');
    if ismember(type, {'c'})
        printf('h.meta[0] = meta[0];\n');
        printf('h.meta[1] = meta[1];\n');
    end
    printf('return h;\n');
    printf('}\n');

end

return;

%***********************************************************************************************************************

function SizeFuncs(p, printf, hFlags)

% called for arrays, textures, and layers
% hFlags = true values for 'handle?'

if nargin < 3, hFlags = [false true]; end

for handle = hFlags

    p = GetParamInfo(p, handle);

    for ed = 1 : p.ned

        printf('%s int %s_s%s%u(%s) {\n', p.attr, p.name, p.suffix, ed - 1, p.param2);

        printf('return %s;\n', GetEDSize(p, ed));

        printf('}\n');

    end

end

return;

%***********************************************************************************************************************

function CoordFuncs(p, printf)

% called for layers only (to get coordinates of current synapse)

p = GetParamInfo(p, false);

for ed = 1 : p.ned

    printf('%s int %s_c%s%u(%s, unsigned int y, unsigned int x) {\n', p.attr, p.name, p.suffix, ed - 1, p.param2);

    printf('int '); BuildED(p, printf, ed);
    
    printf('return c%u;\n', ed - 1);

    printf('}\n');

end

return;

%***********************************************************************************************************************

function InitFunc(p, printf)

% called for layers only

p = GetParamInfo(p, false);

printf('%s void %s_ic(%s, unsigned int y, unsigned int x', p.attr, p.name, p.param2);
printf(', int &c%u', 0 : p.ned - 1);
printf(') {\n');

for ed = 1 : p.ned
    BuildED(p, printf, ed);
end

printf('}\n');

return;

%***********************************************************************************************************************

function LookupFuncs(p, printf, type, varargin)

% type = (a)rray, (t)exture, (c)ommon, (n)euron, or public (w)

% Coordinate transformation plus lookup.

for handle = [false true]

    p = GetParamInfo(p, handle, type, true);

    printf('%s float %s_lk%s(', p.attr, p.name, p.suffix);
    printf('%s', p.param);
    printf(', int c%u', 0 : p.ned - 1);
    printf(') {\n');

    printf('unsigned int '); BuildID(p, printf, 1);
    printf('unsigned int '); BuildID(p, printf, 2);
    if ismember(type, {'c', 't'})
        printf('%s += %s;\n', p.ivars(1), p.meta{1});
        printf('%s += %s;\n', p.ivars(2), p.meta{2});
    end

    DoLookup(p, printf, type, false, varargin{:});

    printf('}\n');

end

% Coordinate transformation only.

if ismember(type, {'a', 'n', 'w'})
    hFlags = [true];
else
    hFlags = [false true];
end

for handle = hFlags

    p = GetParamInfo(p, handle, type);

    printf('%s void %s_gc%s(', p.attr, p.name, p.suffix);
    if ismember(type, {'c', 't'})
        % Needed because we're going to add offsets now.
        printf('%s', p.param);
    else
        printf('%s', p.param2);
    end
    printf(', int c%u', 0 : p.ned - 1);
    printf(', int &y, int &x) {\n');

    BuildID(p, printf, 1);
    BuildID(p, printf, 2);
    if ismember(type, {'c', 't'})
        printf('%s += %s;\n', p.ivars(1), p.meta{1});
        printf('%s += %s;\n', p.ivars(2), p.meta{2});
    end

    printf('}\n');

end

% Lookup only.

p = GetParamInfo(p, true, type, true);

printf('%s float %s_lc(', p.attr, p.name);
if ismember(type, {'a', 'n', 'w'})
    printf('%s, ', p.param1);
elseif ismember(type, {'c', 't'}) && strcmp(p.platform, 'cpu')
    % For the texture-based types we've already added offsets, but may still need a session pointer.
    printf('_Session *s, ');
end
printf('int y, int x) {\n');

DoLookup(p, printf, type, false, varargin{:});

printf('}\n');

% Automatic lookup (from current cell or via synapse).

if ismember(type, {'c', 'n', 'w'})

    p = GetParamInfo(p, false, type, true);

    printf('%s float %s_ln(%s, int y, int x) {\n', p.attr, p.name, p.param1);

    DoLookup(p, printf, type, true, varargin{:});

    printf('}\n');

end

return;

%-----------------------------------------------------------------------------------------------------------------------

function DoLookup(p, printf, type, addOffs, varargin)

if ismember(type, {'a'})

    printf('unsigned int start = %s + (%s << 16);\n', p.meta{1}, p.meta{2});
    printf('return base[start + x * %s[%u] + y];\n', p.siz2p, p.a4 - 1);

elseif ismember(type, {'n', 'w'})

    printf('unsigned int start = %s + (%s << 16);\n', p.meta{1}, p.meta{2});
    printf('return base[start + (%u * %s[-5] + x) * %s[%u] + y];\n', ...
        varargin{1} - 1, p.siz2p, p.siz2p, p.a4 - 1);

else

    if addOffs
        printf('y += %s;\n', p.meta{1});
        printf('x += %s;\n', p.meta{2});
    end

    switch p.platform
    case 'cuda', printf('return tex2D(%s, y, x);\n', varargin{1});
    case 'cpu' , printf('return s->%s.buf[x * s->%s.h + y];\n', varargin{1}, varargin{1});
    end

end

return;

%***********************************************************************************************************************

function SelectFunc(p, printf, iter)

% called for layers only

sel = false(1, p.ned);
sel(p.edims(iter)) = true;

p = GetParamInfo(p, false);

printf('%s void %s_sc(%s', p.attr, p.name, p.param2);
printf(', unsigned int &y, unsigned int &x, int &sc');
for ed = 1 : p.ned
    if sel(ed)
        printf(', int &d%u', ed - 1);
    else
        printf(', int c%u', ed - 1);
    end
end
printf(', int c%u', find(sel) - 1);
printf(') {\n');

for ed = find(sel)
    printf('d%u = c%u;\n', ed - 1, ed - 1);
end

BuildID(p, printf, 1);
BuildID(p, printf, 2);

printf('sc = -1;\n');

printf('}\n');

return;

%***********************************************************************************************************************

function WriteFunc(p, printf)

% called for (c)ommon only

p = GetParamInfo(p, false, 'c');

printf('%s void %s_wn(float *ptr, unsigned int h, %s, int y, int x, float v) {\n', p.attr, p.name, p.param1);

printf('y += %s;\n', p.meta{1});
printf('x += %s;\n', p.meta{2});

printf('ptr[x * h + y] = v;\n');

printf('}\n');

return;

%***********************************************************************************************************************

function p = GetParamInfo(p, handle, type, lookup)

% handle = true or false
% type   = (a)rray, (t)exture, (c)ommon, (n)euron, or public (w)
% lookup = are we actually going to DO the lookup?

% p.suffix  = 'h' if a handle-based function, '' if not
% p.param   = parameters needed for lookup and coordinate transform
% p.param1  = parameters needed for lookup
% p.param2  = parameters needed for coordinate transform
% p.siz2p   = base reference to siz2p
% p.meta{1} = reference to 1st 16-bit lookup value
% p.meta{2} = reference to 2nd 16-bit lookup value

if nargin < 3, type   = ''   ; end
if nargin < 4, lookup = false; end

if handle
    p.suffix = 'h';
    p.siz2p  = 'h.siz2p'; % cast to unsigned int?
    p.param  = sprintf('%s_h h', p.name);
    p.param1 = p.param;
    p.param2 = p.param;
else
    p.suffix = '';
    p.siz2p  = 'siz2p'; % cast to unsigned int?
    if ismember(type, {'c'})
        p.param  = 'const unsigned short *meta, const unsigned short *siz2p';
        p.param1 = 'const unsigned short *meta';
        p.param2 = 'const unsigned short *siz2p';
    else
        p.param  = 'const unsigned short *siz2p';
        p.param1 = p.param;
        p.param2 = p.param;
    end
end

if lookup
    if ismember(type, {'a', 'n', 'w'})
        p.param  = sprintf('const float *base, %s', p.param );
        p.param1 = sprintf('const float *base, %s', p.param1);
    elseif ismember(type, {'c', 't'}) && strcmp(p.platform, 'cpu')
        p.param  = sprintf('_Session *s, %s', p.param );
        p.param1 = sprintf('_Session *s, %s', p.param1);
    end
end

if ismember(type, {'c'})
    if handle
        meta = 'h.meta'; % cast to unsigned int?
    else
        meta = 'meta'; % cast to unsigned int?
    end
    p.meta{1} = sprintf('%s[0]', meta);
    p.meta{2} = sprintf('%s[1]', meta);
elseif ismember(type, {'w'})
    p.meta{1} = sprintf('%s[-4]', p.siz2p);
    p.meta{2} = sprintf('%s[-3]', p.siz2p);
else
    p.meta{1} = sprintf('%s[-2]', p.siz2p);
    p.meta{2} = sprintf('%s[-1]', p.siz2p);
end

return;

%***********************************************************************************************************************

function BuildED(p, printf, ed)

% Generates code for building an external dimension from internal dimensions.

cs = find(p.edims == ed);
nep = numel(cs);

printf('c%u = %s;\n', ed - 1, GetIP(p, cs(nep)));

for i = nep - 1 : -1 : 1
    printf('c%u <<= %s[%u];\n', ed - 1, p.siz2p, cs(i) - 1);
    printf('c%u += %s;\n', ed - 1, GetIP(p, cs(i)));
end

return;

%***********************************************************************************************************************

function BuildID(p, printf, id)

% Generates code for building an internal dimension from external dimensions.

cs = find(p.idims == id);
[ans, inds] = sort(p.iparts(cs));
cs = cs(inds);
nip = numel(cs);

ds  = cs;
pow = p.pow(cs);
ds (cs == p.ac) = p.a3;
pow(cs == p.ac) = false;

if nip == 0

    printf('%s = 0;\n', p.ivars(id));

else

    printf('%s = %s;\n', p.ivars(id), GetEP(p, cs(nip)));

    for i = nip - 1 : -1 : 1
        if pow(i)
            printf('%s <<= %s[%u];\n', p.ivars(id), p.siz2p, ds(i) - 1);
        else
            printf('%s *= %s[%u];\n', p.ivars(id), p.siz2p, ds(i) - 1);
        end
        printf('%s += %s;\n', p.ivars(id), GetEP(p, cs(i)));
    end

end

return;

%***********************************************************************************************************************

function s = GetEP(p, c)

% Generates an expression for extracting one part of an external dimension.

ed = p.edims(c);
cs = find(p.edims == ed);
ep = find(cs == c);
nep = numel(cs);

s = sprintf('c%u', ed - 1);

if ep > 1

    s = sprintf('(%s >> (', s);

    op = '';
    for i = 1 : ep - 1
        s = sprintf('%s%s%s[%u]', s, op, p.siz2p, cs(i) - 1);
        op = ' + ';
    end
    
    s = sprintf('%s))', s);

end

if ep < nep

    s = sprintf('(%s & ((1 << %s[%u]) - 1))', s, p.siz2p, cs(ep) - 1);

end

if p.shift(c)
    s = sprintf('_Roll(%s, %s[%u], %s[%u])', s, p.siz2p, p.rc - 1, p.siz2p, c - 1);
end

return;

%***********************************************************************************************************************

function s = GetIP(p, c)

% Generates an expression for extracting one part of an internal dimension.

id = p.idims (c);
ip = p.iparts(c);
cs = find(p.idims == id);
[ans, inds] = sort(p.iparts(cs));
cs = cs(inds);
nip = numel(cs);

pow = p.pow(cs);
pow(cs == p.ac) = false;
cs (cs == p.ac) = p.a3;

s = p.ivars(id);

is = find(pow(1 : ip - 1));
if ~isempty(is)

    s = sprintf('(%s >> (', s);

    op = '';
    for i = is
        s = sprintf('%s%s%s[%u]', s, op, p.siz2p, cs(i) - 1);
        op = ' + ';
    end

    s = sprintf('%s))', s);

end

is = find(~pow(1 : ip - 1));
if ~isempty(is)

    s = sprintf('(%s / (', s);

    op = '';
    for i = is
        s = sprintf('%s%s%s[%u]', s, op, p.siz2p, cs(i) - 1);
        op = ' * ';
    end

    s = sprintf('%s))', s);

end

if ip < nip

    if pow(ip)
        s = sprintf('(%s & ((1 << %s[%u]) - 1))', s, p.siz2p, cs(ip) - 1);
    else
        s = sprintf('(%s %% %s[%u])', s, p.siz2p, cs(ip) - 1);
    end

end

if p.shift(c)
    s = sprintf('_Unroll(%s, %s[%u], %s[%u])', s, p.siz2p, p.rc - 1, p.siz2p, c - 1);
end

return;

%***********************************************************************************************************************

function s = GetEDSize(p, ed)

% Generates an expression for computing the size of an external dimension.

cs = find(p.edims == ed);

s = sprintf('%s[%u]', p.siz2p, cs(end) - 1);

if numel(cs) > 1

    s = sprintf('(%s << (', s);

    op = '';
    for i = 1 : numel(cs) - 1
        s = sprintf('%s%s%s[%u]', s, op, p.siz2p, cs(i) - 1);
        op = ' + ';
    end
    
    s = sprintf('%s))', s);

end

return;