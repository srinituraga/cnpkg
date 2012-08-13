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
p.eparts = [];
p.shift  = false(0);
for ed = 1 : numel(d.dims)
    for ep = 1 : numel(d.dims{ed})
        p.idims (end + 1) = d.dims  {ed}(ep);
        p.iparts(end + 1) = d.dparts{ed}(ep);
        p.edims (end + 1) = ed;
        p.eparts(end + 1) = ep;
        p.shift (end + 1) = (d.dmap(ed) == 2);
    end
end

% Index of the first 'y' coordinate.  This gets padded to siz3a(1).
p.ac = find((p.idims == 1) & (p.iparts == 1));

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
case 'lookup', LookupFuncs(p, printf, varargin{:});
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

    printf('%s int %s_s%s(%s, unsigned int d) {\n', p.attr, p.name, p.suffix, p.param2);

    printf('return (int)%s[d];\n', p.siz2p);

    printf('}\n');

end

return;

%***********************************************************************************************************************

function CoordFuncs(p, printf)

% called for layers only

p = GetParamInfo(p, false);

for c = 1 : numel(p.edims)

    printf('%s int %s_c%s%u(%s, unsigned int y, unsigned int x) {\n', p.attr, p.name, p.suffix, c - 1, p.param2);

    s = p.ivars(p.idims(c));
    if p.iparts(c) < max(p.iparts(p.idims == p.idims(c)))
        s = sprintf('(%s %% %s)', s, ProdSiz2(p, find((p.idims == p.idims(c)) & (p.iparts <= p.iparts(c)))));
    end
    if p.iparts(c) > 1
        s = sprintf('(%s / %s)' , s, ProdSiz2(p, find((p.idims == p.idims(c)) & (p.iparts <  p.iparts(c)))));
    end

    if p.shift(c)
        s = sprintf('_Unroll(%s, %s[%u], %s[%u])', s, ...
            p.siz2p, numel(p.edims) + 2, p.siz2p, c - 1);
    end

    printf('return (int)%s;\n', s);

    printf('}\n');

end

return;

%***********************************************************************************************************************

function LookupFuncs(p, printf, type, varargin)

% type = (a)rray, (t)exture, (c)ommon, (n)euron, or public (w)

% Coordinate transformation plus lookup.

for handle = [false true]

    p = GetParamInfo(p, handle, type, true);

    printf('%s float %s_lk%s(', p.attr, p.name, p.suffix);
    printf('%s', p.param);
    printf(', int c%u', 0 : numel(p.edims) - 1);
    printf(') {\n');

    BuildYX(p, printf, type, 1, true);
    BuildYX(p, printf, type, 2, true);

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
    printf(', int c%u', 0 : numel(p.edims) - 1);
    printf(', int &y, int &x) {\n');

    BuildYX(p, printf, type, 1, false);
    BuildYX(p, printf, type, 2, false);

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

function BuildYX(p, printf, type, id, dec)

if any(p.idims == id)

    nip = max(p.iparts(p.idims == id));

    for ip = nip : -1 : 1

        c = find((p.idims == id) & (p.iparts == ip));

        s = sprintf('c%u', c - 1);

        if p.shift(c)
            s = sprintf('_Roll(%s, %s[%u], %s[%u])', s, ...
                p.siz2p, numel(p.edims) + 2, p.siz2p, c - 1);
        end

        if ip == nip
            if dec, printf('unsigned int '); end
            printf('%s = %s;\n', p.ivars(id), s);
        else
            if c == p.ac, c = numel(p.edims) + 1; end
            printf('%s = %s * %s[%u] + %s;\n', p.ivars(id), p.ivars(id), p.siz2p, c - 1, s);
        end

    end

else

    if dec, printf('unsigned int '); end
    printf('%s = 0;\n', p.ivars(id));

end

if ismember(type, {'c', 't'})
    printf('%s += %s;\n', p.ivars(id), p.meta{id});
end

return;

%-----------------------------------------------------------------------------------------------------------------------

function DoLookup(p, printf, type, addOffs, varargin)

if ismember(type, {'a'})

    printf('unsigned int start = %s + (%s << 16);\n', p.meta{1}, p.meta{2});
    printf('return base[start + x * %s[%u] + y];\n', p.siz2p, numel(p.edims) + 1);

elseif ismember(type, {'n', 'w'})

    printf('unsigned int start = %s + (%s << 16);\n', p.meta{1}, p.meta{2});
    printf('return base[start + (%u * %s[-5] + x) * %s[%u] + y];\n', ...
        varargin{1} - 1, p.siz2p, p.siz2p, numel(p.edims) + 1);

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
    p.siz2p  = '(unsigned int)h.siz2p';
    p.param  = sprintf('%s_h h', p.name);
    p.param1 = p.param;
    p.param2 = p.param;
else
    p.suffix = '';
    p.siz2p  = '(unsigned int)siz2p';
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
        meta = '(unsigned int)h.meta';
    else
        meta = '(unsigned int)meta';
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

function s = ProdSiz2(p, cs)

cs(cs == p.ac) = numel(p.edims) + 1;

s = sprintf('%s[%u]', p.siz2p, cs(1) - 1);

if numel(cs) > 1

    for c = cs(2 : end)
        s = sprintf('%s * %s[%u]', s, p.siz2p, c - 1);
    end

    s = sprintf('(%s)', s);

end

return;