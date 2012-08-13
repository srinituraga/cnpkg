function def = cns_def(m)

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

if ischar(m)
    def = GetDef(m);
else
    def = GetModelDef(m);
end

return;

%***********************************************************************************************************************

function def = GetModelDef(m)

d = GetDef(m.package);

def = rmfield(d, {'types', 'type'});

if isfield(m, 'layers'), numLayers = numel(m.layers); else numLayers = 0; end
if isfield(m, 'groups'), numGroups = numel(m.groups); else numGroups = 0; end

def.layers = cell(1, numLayers);

gs = zeros(1, numLayers);
ts = zeros(1, numLayers);

for z = 1 : numLayers

    name = m.layers{z}.type;
    if ~isfield(d.type, name)
        error('z=%u: type name "%s" is invalid', z, name);
    end
    if d.type.(name).abstract
        error('z=%u: type "%s" is abstract', z, name);
    end

    def.layers{z} = d.type.(name);
    def.layers{z}.type = name;

    if isfield(m.layers{z}, 'groupNo'), gs(z) = m.layers{z}.groupNo; end
    ts(z) = def.layers{z}.typeNo;

end

gCount = max([gs 0]);
if numel(unique(gs(gs > 0))) ~= gCount
    error('group numbers must be contiguous');
end
if numGroups ~= gCount
    error('"groups" must have %u elements', gCount);
end

def.groups = cell(1, gCount + sum(gs == 0));
def.gCount = gCount;

for g = 1 : gCount

    zs = find(gs == g);
    if ~isscalar(unique(ts(zs))), error('all layers in a group must be of the same type'); end

    def.groups{g} = struct;
    def.groups{g}.zs = zs;

    for z = zs
        def.layers{z}.g    = g;
        def.layers{z}.auto = false;
    end

end

g = gCount;

for z = find(gs == 0)

    g = g + 1;

    def.groups{g} = struct;
    def.groups{g}.zs = z;

    def.layers{z}.g    = g;
    def.layers{z}.auto = true;

end

return;

%***********************************************************************************************************************

function def = GetDef(package)

persistent Cache;
if isempty(Cache), Cache = struct; end

if strcmp(package, '%clear%')
    Cache = struct;
    def   = struct;
    return;
end

if isfield(Cache, package)
    def = Cache.(package);
    return;
end

path = fileparts(which(package));
if isempty(path), error('cannot find definition function "%s.m"', package); end
saveFile = fullfile(path, [package '_compiled_def.mat']);

if ~exist(saveFile, 'file')
    error('package "%s" has not been compiled; see cns_build', package);
end

load(saveFile, 'def');

Cache.(package) = def;

return;