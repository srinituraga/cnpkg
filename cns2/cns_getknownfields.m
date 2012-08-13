function [k, u] = cns_getknownfields(m)

% CNS_GETKNOWNFIELDS
%    Click <a href="matlab: cns_help('cns_getknownfields')">here</a> for help.

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

def = cns_def(m);

mm = m;
if isfield(m, 'layers'), mm = rmfield(mm, 'layers'); end
if isfield(m, 'groups'), mm = rmfield(mm, 'groups'); end
[k, u] = Separate(mm, cns_reservednames(true), def, []);

for z = 1 : numel(def.layers)
    [k.layers{z}, u.layers{z}] = Separate(m.layers{z}, cns_reservednames(false), def.layers{z}, false);
end

for g = 1 : def.gCount
    z = def.groups{g}.zs(1);
    [k.groups{g}, u.groups{g}] = Separate(m.groups{g}, cns_reservednames(false), def.layers{z}, true);
end

return;

%***********************************************************************************************************************

function [k, u] = Separate(m, reserved, d, group)

if isequal(group, false) && d.auto, group = []; end

known = reserved;

for i = 1 : numel(d.syms)
    name = d.syms{i};
    if d.sym.(name).field && (isempty(group) || (d.sym.(name).group == group))
        known{end + 1} = name;
    end
end

names = fieldnames(m);

k = struct;

for i = 1 : numel(known)
    if ismember(known{i}, names)
        k.(known{i}) = m.(known{i});
    end
end

u = struct;

for i = 1 : numel(names)
    if ~ismember(names{i}, known)
        u.(names{i}) = m.(names{i});
    end
end
    
return;