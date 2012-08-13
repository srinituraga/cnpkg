function cns_dump(dump, mode, varargin)

% CNS_DUMP
%    Click <a href="matlab: cns_help('cns_dump')">here</a> for help.

%***********************************************************************************************************************

% Copyright (C) 2010 by Jim Mutch (www.jimmutch.com).
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

if nargin < 2, mode = 'summary'; end

switch mode
case 'summary', Summary(dump.G, dump.G.s, dump.h, dump.G.m, dump.G.def, varargin{:});
case 'textile', TexTile(dump.G, dump.G.s, dump.h, dump.G.m, dump.G.def, varargin{:});
otherwise     , error('invalid mode');
end

return;

%***********************************************************************************************************************

function Summary(G, s, h, m, def)

fprintf('CONSTANT MEMORY:\n');
fprintf('  constants: %u bytes\n', numel(h.cData) * 4);
fprintf('  metadata: %u bytes\n', numel(h.cMeta) * 2);

fprintf('GLOBAL MEMORY:\n');

for i = 1 : def.tCount
    % TODO: overhead
    fprintf('  texture "%s" (%ux%u): %u bytes\n', def.tList{i}, ...
        size(h.tTData{i}), numel(h.tTData{i}) * 4);
end

for i = 1 : def.ccCount
    total = numel(h.tCData{i});
    [cells, align] = TexUsage(s, m, def, def.ccList{i}, i);
    pack  = round(100 * (total - align) / total);
    align = round(100 * (align - cells) / total);
    fprintf('  texture "%s" (%ux%u): %u bytes\n', def.ccList{i}, ...
        size(h.tCData{i}), total * 4);
    fprintf('    (overhead: %u%% align, %u%% pack)\n', align, pack);
end

for i = 1 : def.cvCount
    total = numel(h.tVData{i});
    [cells, align] = TexUsage(s, m, def, def.cvList{i}, i);
    pack  = round(100 * (total - align) / total);
    align = round(100 * (align - cells) / total);
    if s.independent
        fprintf('  texture "%s" (%ux%u): %u bytes\n', def.cvList{i}, ...
            size(h.tVData{i}), total * 4);
    else
        fprintf('  texture "%s" (%ux%u, double buffered): %u bytes\n', def.cvList{i}, ...
            size(h.tVData{i}), total * 4 * 2);
    end
    fprintf('    (overhead: %u%% align, %u%% pack)\n', align, pack);
end

fprintf('  explicit synapse data: %u bytes\n', numel(h.dNeurons) * 4 + numel(h.dSynapses) * 2);
fprintf('  other data: %u bytes\n', numel(h.dData) * 4 + numel(h.dWData) * 4);
fprintf('  block table: %u bytes\n', numel(h.dBlocks) * 2);

return;

%***********************************************************************************************************************

function TexTile(G, s, h, m, def, field)

% TODO: support N-D array textures

resNo = [];
if isempty(resNo), resNo = find(strcmpi(def.ccList, field)); end
if isempty(resNo), resNo = find(strcmpi(def.cvList, field)); end
if isempty(resNo), error('cannot find texture "%s"', field); end

[zs, ans, ySizes, xCounts, yOffs, xOffs] = TexLayerInfo(s, m, def, field, resNo);

xs = zeros(4, numel(zs));
ys = zeros(4, numel(zs));
cs = zeros(1, numel(zs));
for i = 1 : numel(zs)
    xs(:, i) = [0 0 1 1]' * xCounts(i) + xOffs(i);
    ys(:, i) = [0 1 1 0]' * ySizes (i) + yOffs(i);
    cs(1, i) = i;
end

cla;
patch(xs, ys, cs);
axis ij tight;

return;

%***********************************************************************************************************************

function [cells, align] = TexUsage(s, m, def, field, resNo)

[ans, cells, ySizes, xCounts] = TexLayerInfo(s, m, def, field, resNo);

cells = sum(cells);
align = sum(ySizes .* xCounts);

return;

%***********************************************************************************************************************

function [zs, cells, ySizes, xCounts, yOffs, xOffs] = TexLayerInfo(s, m, def, field, resNo)

[ans, field] = strtok(field, '_');
field = field(2 : end);

zs      = [];
cells   = [];
ySizes  = [];
xCounts = [];
yOffs   = [];
xOffs   = [];

for z = 1 : numel(s.layers)
    if isfield(def.layers{z}.sym, field) && (def.layers{z}.sym.(field).resNo == resNo)

        pos = s.layers(z).tOff + 2 * (def.layers{z}.sym.(field).pos - 1);

        zs     (end + 1) = z;
        cells  (end + 1) = prod(cellfun(@prod, m.layers{z}.size));
        ySizes (end + 1) = s.layers(z).ySize;
        xCounts(end + 1) = s.layers(z).xCount;
        yOffs  (end + 1) = s.cMeta(pos + 1);
        xOffs  (end + 1) = s.cMeta(pos + 2);

    end
end

return;