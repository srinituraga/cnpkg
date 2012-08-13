function varargout = cns_package(method, m, varargin)

% CNS_PACKAGE
%    Click <a href="matlab: cns_help('cns_package')">here</a> for help.

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
elseif ischar(m)
    m = struct('package', {m});
else
    error('invalid m');
end

def = cns_def(m);

if ~ismember(method, def.methods), error('invalid method'); end

global CNS_METHOD;
save_method = CNS_METHOD;
CNS_METHOD = [];

err = [];
try
    [varargout{1 : nargout}] = feval([m.package '_cns'], method, m, varargin{:});
catch
    err = lasterror;
end

if isempty(save_method)
    clear global CNS_METHOD;
else
    CNS_METHOD = save_method;
end

if ~isempty(err), rethrow(err); end

return;