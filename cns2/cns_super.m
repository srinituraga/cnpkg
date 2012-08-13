function varargout = cns_super(varargin)

% CNS_SUPER
%    Click <a href="matlab: cns_help('cns_super')">here</a> for help.

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

stack = dbstack;
if numel(stack) < 2, error('not in a method call'); end
[cname, method] = strtok(stack(2).name, '.');
if isempty(cname) || (numel(method) < 2), error('not in a method call'); end
method = method(2 : end);
pos = find(cname == '_', 1, 'last');
if isempty(pos) || (pos == 1) || (pos == numel(cname)), error('not in a CNS method call'); end
package = cname(1 : pos - 1);
type    = cname(pos + 1 : end);

if strcmp(type, 'base')
    sname = 'cns_base';
else
    list = superclasses(cname); % Assumes a parent will come before its ancestors.
    pos = strmatch([package '_'], list);
    if isempty(pos), error('no parent cell type'); end % Should not happen.
    sname = list{pos(1)};
end

[varargout{1 : nargout}] = feval([sname '.' method], varargin{:});

return;