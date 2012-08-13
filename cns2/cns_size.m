function s = cns_size(a, dim)

% CNS_SIZE
%    Click <a href="matlab: cns_help('cns_size')">here</a> for help.

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

if ~isscalar(dim) || (dim < 1) || (mod(dim, 1) ~= 0)
    error('invalid number of dimensions');
end

s = size(a);

if numel(s) < dim
    s(end + 1 : dim) = 1;
elseif numel(s) > dim
    if (dim == 1) && (numel(s) == 2)
        if all(s ~= 1) && any(s > 0), error('invalid size'); end
        s = prod(s);
    else
        error('invalid size');
    end
end

s = num2cell(s);

return;
