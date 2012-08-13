function [dno, dname, dmap] = cns_finddim(m, z, dim, mapped)

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

if (nargin < 4) || isempty(mapped)
    mapped = [0 1 2];
end

def = cns_def(m);

if isnumeric(dim)
    if (dim < 1) || (dim > numel(def.layers{z}.dims)) || (mod(dim, 1) ~= 0), error('invalid dimension number'); end
    dname = def.layers{z}.dnames{dim};
    dno = dim;
else
    if isempty(dim), error('invalid dimension name'); end
    [ans, dno] = ismember(dim, def.layers{z}.dnames);
    if dno == 0, error('layer %u does not have dimension "%s"', z, dim); end
    dname = dim;
end

dmap = def.layers{z}.dmap(dno);

if ~ismember(dmap, mapped)
    error('dimension "%s" in layer %u has the wrong mapping', dname, z);
end

return;