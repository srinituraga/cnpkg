function varargout = cns_iconv(m, z, varargin)

% CNS_ICONV
%    Click <a href="matlab: cns_help('cns_iconv')">here</a> for help.

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

if isstruct(m)
    siz = cellfun(@prod, m.layers{z}.size);
elseif isnumeric(m) && isempty(z)
    siz = m;
else
    error('invalid m');
end

ni = numel(varargin);
no = max(nargout, 1);

if (ni < 1) || (ni > numel(siz)), error('incorrect number of arguments'); end
if no > numel(siz), error('incorrect number of outputs'); end

si = [siz(1 : ni - 1), prod(siz(ni : end))];
so = [siz(1 : no - 1), prod(siz(no : end))];

for i = 1 : ni
    if any(varargin{i}(:) < 1    ), error('index out of range'); end
    if any(varargin{i}(:) > si(i)), error('index out of range'); end
end

if ni == no
    varargout = varargin;
elseif ni == 1
    [varargout{1 : no}] = ind2sub(so, varargin{1});
elseif no == 1
    varargout{1} = sub2ind(si, varargin{:});
else
    [varargout{1 : no}] = ind2sub(so, sub2ind(si, varargin{:}));
end

return;