function varargout = cns_findnearest(m, z, dim, c, pz, n)

% CNS_FINDNEAREST
%    Click <a href="matlab: cns_help('cns_findnearest')">here</a> for help.

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

[ans, dname] = cns_finddim(m, z, dim, [1 2]);

[varargout{1 : nargout}] = cns_findnearest_at(m, pz, dname, cns_center(m, z, dname, c), n);

return;