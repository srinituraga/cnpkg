function r = cns_istype(m, n, theType)

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

if cns_isstruct(m)

    package = m.package;

    if n == 0, error('invalid layer or group number'); end
    if n > 0
        type = m.layers{n}.type;
    else
        z = cns_grouplayers(m, -n, 1);
        if isempty(z), error('invalid group number'); end
        type = m.layers{z}.type;
    end

elseif ischar(m)

    package = m;
    type    = n;

else

    error('invalid model or package name');

end

cname = [package '_' type];
if isempty(which(cname)), error('"%s.m" not found', cname); end

if strcmp(type, theType)
    r = true;
else
    r = ismember([package '_' theType], superclasses(cname));
end

return;