function varargout = cns_findnearest_at(m, pz, dim, p, n)

% CNS_FINDNEAREST_AT
%    Click <a href="matlab: cns_help('cns_findnearest_at')">here</a> for help.

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

if (nargout < 2) || (nargout > 5), error('invalid number of outputs'); end

[dno, dname] = cns_finddim(m, pz, dim, [1 2]);

t  = m.layers{pz}.size{dno};
s  = m.layers{pz}.([dname '_start']);
dd = 1 / m.layers{pz}.([dname '_space']);

c1 = ceil((p - s) * dd - 0.5 * n - 0.001);
c2 = c1 + n - 1;

v1 = min(max(c1,  0), t    );
v2 = min(max(c2, -1), t - 1);

varargout{1} = v1 + 1;
varargout{2} = v2 + 1;
if nargout == 3
    varargout{3} = (v1 == c1) & (v2 == c2);
elseif nargout >= 4
    varargout{3} = c1 + 1;
    varargout{4} = c2 + 1;
    if nargout == 5
        varargout{5} = (v1 <= v2);
    end
end

return;