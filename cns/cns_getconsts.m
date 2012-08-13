function varargout = cns_getconsts(m, n, varargin)

% CNS_GETCONSTS
%    Click <a href="matlab: cns_help('cns_getconsts')">here</a> for help.

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

if isempty(n), error('invalid n'); end

args = varargin;

if ~isempty(args) && isstruct(args{1})
    s   = args{1};
    str = true;
    args = args(2 : end);
else
    s   = struct;
    str = false;
end

if isempty(args)
    if nargout > 1, error('incorrect number of outputs'); end
    all = true;
    str = true;
elseif (numel(args) == 1) && iscell(args{1})
    if nargout > 1, error('incorrect number of outputs'); end
    names = args{1};
    all = false;
    str = true;
elseif ~str
    if max(nargout, 1) ~= numel(args), error('incorrect number of outputs'); end
    names = args;
    all = false;
else
    error('invalid input');
end

def = cns_def(m);

c = GetConsts(struct, def);
if n > 0
    c = GetConsts(c, def.layers{n});
elseif n < 0
    z = def.groups{-n}.zs(1);
    c = GetConsts(c, def.layers{z});
end

if all, names = fieldnames(c); end
for i = 1 : numel(names)
    if ~isfield(c, names{i}), error('constant name "%s" is invalid', names{i}); end
    s.(names{i}) = c.(names{i});
end

if str
    varargout{1} = s;
else
    varargout = struct2cell(s);
end

return;

%***********************************************************************************************************************

function c = GetConsts(c, d)

for i = 1 : numel(d.cat.c.syms)
    name = d.cat.c.syms{i};
    c.(name) = d.sym.(name).value;
end

return;