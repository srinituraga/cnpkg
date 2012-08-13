function r = cns_istype(m, n, type)

% CNS_ISTYPE
%    Click <a href="matlab: cns_help('cns_istype')">here</a> for help.

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

if isstruct(m)
elseif ischar(m)
    if ~ischar(n), error('invalid n'); end
    m = struct('package', {m});
else
    error('invalid m');
end

if isnumeric(n)
    def = cns_def(m);
    if n < 0
        n = -n;
        if n > def.gCount, error('invalid n'); end
        d = def.layers{def.groups{n}.zs(1)};
    else
        d = def.layers{n};
    end
elseif ischar(n)
    if isempty(n), error('invalid n'); end
    def = cns_def(m.package);
    d = def.type.(n);
else
    error('invalid n');
end

r = ismember(type, d.typePath);

return;