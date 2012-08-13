function m = cns_getdflts(m, n)

% CNS_GETDFLTS
%    Click <a href="matlab: cns_help('cns_getdflts')">here</a> for help.

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

if nargin < 2, n = []; end

def = cns_def(m);

if isempty(n) || (n == 0)
    m = GetDflts(m, def, []);
end

for z = 1 : numel(def.layers)
    if isempty(n) || (n == z)
        m.layers{z} = GetDflts(m.layers{z}, def.layers{z}, false);
    end
end

for g = 1 : def.gCount
    if isempty(n) || (n == -g)
        z = def.groups{g}.zs(1);
        m.groups{g} = GetDflts(m.groups{g}, def.layers{z}, true);
    end
end

return;

%***********************************************************************************************************************

function m = GetDflts(m, d, group)

if isequal(group, false) && d.auto, group = []; end

for i = 1 : numel(d.syms)
    name = d.syms{i};
    if d.sym.(name).field && (isempty(group) || (d.sym.(name).group == group)) && ~isfield(m, name)
        dflt = d.sym.(name).value;
        if ~isequalwithequalnans(dflt, NaN), m.(name) = dflt; end
    end
end

return;