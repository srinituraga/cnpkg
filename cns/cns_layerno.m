function r = cns_layerno(m, name, err)

% CNS_LAYERNO
%    Click <a href="matlab: cns_help('cns_layerno')">here</a> for help.

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
if isfield(m, 'layers')
    numLayers = numel(m.layers);
else
    numLayers = 0;
end

if nargin < 2

    r = struct;
    for z = 1 : numLayers
        if isfield(m.layers{z}, 'name')
            name = m.layers{z}.name;
            if isvarname(name)
                r.(name) = z;
            end
        end
    end

elseif isnumeric(name) && isscalar(name)

    r = name;

    if (r < 1) || (r > numLayers) || (mod(r, 1) ~= 0)
        if err
            error('invalid layer number: %s', num2str(r));
        else
            r = 0;
        end
    end

elseif ischar(name)

    r = 0;
    for z = 1 : numLayers
        if isfield(m.layers{z}, 'name') && strcmp(m.layers{z}.name, name)
            r = z;
            break;
        end
    end

    if (r == 0) && err
        error('invalid layer name: "%s"', name);
    end

else

    error('invalid layer');

end

return;