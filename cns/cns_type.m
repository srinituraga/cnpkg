function varargout = cns_type(method, m, n, varargin)

% CNS_TYPE
%    Click <a href="matlab: cns_help('cns_type')">here</a> for help.

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
    if ~ischar(n), error('invalid n'); end
    m = struct('package', {m});
else
    error('invalid m');
end

if isnumeric(n)
    def = cns_def(m);
    if n < 0
        n = -n;
        if n > def.gCount, error('invalid n'); end
        d = def.layers{def.groups{n}.zs(1)};
    else
        d = def.layers{n};
    end
    type = d.type;
elseif ischar(n)
    if isempty(n), error('invalid n'); end
    def = cns_def(m.package);
    type = n;
    n = [];
    d = def.type.(type);
else
    error('invalid n');
end

if ~isfield(d.method, method), error('invalid method'); end
list = d.method.(method);
first = list{end};
rest  = list(1 : end - 1);

global CNS_METHOD;
save_method = CNS_METHOD;
CNS_METHOD = struct;
CNS_METHOD.method = method;
CNS_METHOD.super  = rest;

err = [];
try
    if isempty(n)
        [varargout{1 : nargout}] = feval([m.package '_cns_type_' first], method, m, varargin{:});
    else
        [varargout{1 : nargout}] = feval([m.package '_cns_type_' first], method, m, n, varargin{:});
    end
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