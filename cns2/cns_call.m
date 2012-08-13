function varargout = cns_call(m, n, method, varargin)

% CNS_CALL
%    Click <a href="matlab: cns_help('cns_call')">here</a> for help.

%***********************************************************************************************************************

% Copyright (C) 2011 by Jim Mutch (www.jimmutch.com).
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

    if n == 0, error('invalid layer or group number'); end
    if n > 0
        type = m.layers{n}.type;
    else
        z = cns_grouplayers(m, -n, 1);
        if isempty(z), error('invalid group number'); end
        type = m.layers{z}.type;
    end

    [varargout{1 : nargout}] = feval([m.package '_' type '.' method], m, abs(n), varargin{:});

elseif ischar(m)

    [varargout{1 : nargout}] = feval([m '_' n '.' method], varargin{:});

else

    error('invalid model or package name');

end

return;