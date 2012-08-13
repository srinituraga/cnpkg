function r = cns_isfield(s, fields)

% Internal CNS function.

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

% TODO: need to check all uses of 'isfield' in CNS.

if ~isobject(s)
    r = isfield(s, fields);
    return;
end

if ischar(fields), fields = {fields}; end

r = false(size(fields));

for i = 1 : numel(fields)
    try
        ans = size(s.(fields{i}));
        r(i) = true;
    catch
        r(i) = false;
    end
end

return;