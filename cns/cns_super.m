function varargout = cns_super(m, varargin)

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

global CNS_METHOD;
if isempty(CNS_METHOD), error('not in a method call'); end
if isempty(CNS_METHOD.super), error('no superclass implementation to call'); end
method = CNS_METHOD.method;
next   = CNS_METHOD.super{end};
CNS_METHOD.super = CNS_METHOD.super(1 : end - 1);

[varargout{1 : nargout}] = feval([m.package '_cns_type_' next], method, m, varargin{:});

return;