function names = cns_reservednames(globalDef, memonly)

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

if nargin < 1, globalDef = []   ; end
if nargin < 2, memonly   = false; end

if isempty(globalDef)
    n1 = GlobalNames(false);
    n2 = KernelNames(false);
    names = [n1, n2(~ismember(n2, n1))];
elseif globalDef
    names = GlobalNames(memonly);
else
    names = KernelNames(memonly);
end

return;

%***********************************************************************************************************************

function names = GlobalNames(memonly)

names = {};
mem   = false(0, 0);

names{end + 1} = 'package'    ; mem(end + 1) = true;
names{end + 1} = 'layers'     ; mem(end + 1) = false;
names{end + 1} = 'groups'     ; mem(end + 1) = false;
names{end + 1} = 'independent'; mem(end + 1) = true;
names{end + 1} = 'quiet'      ; mem(end + 1) = true;

if memonly, names = names(mem); end

return;

%***********************************************************************************************************************

function names = KernelNames(memonly)

names = {};
mem   = false(0, 0);

names{end + 1} = 'name'     ; mem(end + 1) = true;
names{end + 1} = 'type'     ; mem(end + 1) = true;
names{end + 1} = 'size'     ; mem(end + 1) = true;
names{end + 1} = 'synapseIs'; mem(end + 1) = false;
names{end + 1} = 'synapseZs'; mem(end + 1) = false;
names{end + 1} = 'synapseTs'; mem(end + 1) = false;
names{end + 1} = 'groupNo'  ; mem(end + 1) = true;
names{end + 1} = 'stepNo'   ; mem(end + 1) = true;

if memonly, names = names(mem); end

return;
