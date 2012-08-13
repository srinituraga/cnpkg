function r = cns_groupno(m, name, err)

% CNS_GROUPNO
%    Click <a href="matlab: cns_help('cns_groupno')">here</a> for help.

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

if nargin < 1, error('not enough arguments'); end
if nargin < 3, err = true; end

if ~isstruct(m), error('invalid model'); end
if isfield(m, 'groups')
    numGroups = numel(m.groups);
else
    numGroups = 0;
end

if nargin < 2

    r = struct;
    for g = 1 : numGroups
        if isfield(m.groups{g}, 'name')
            name = m.groups{g}.name;
            if isvarname(name)
                r.(name) = g;
            end
        end
    end

elseif isnumeric(name) && isscalar(name)

    r = name;

    if (r < 1) || (r > numGroups) || (mod(r, 1) ~= 0)
        if err
            error('invalid group number: %s', num2str(r));
        else
            r = 0;
        end
    end

elseif ischar(name)

    r = 0;
    for g = 1 : numGroups
        if isfield(m.groups{g}, 'name') && strcmp(m.groups{g}.name, name)
            r = g;
            break;
        end
    end

    if (r == 0) && err
        error('invalid group name: "%s"', name);
    end

else

    error('invalid group');

end

return;