function varargout = cns_transparse(mode, varargin)

% Internal CNS function.

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

switch mode
case 'parse', [varargout{1 : nargout}] = Parse(varargin{:});
case 'undef', [varargout{1 : nargout}] = Undef(varargin{:});
case 'iter' , [varargout{1 : nargout}] = Iter (varargin{:});
otherwise   , error('invalid mode');
end

return;

%***********************************************************************************************************************

function d = Parse(d, u, shiftOK)

if ~isfield(u, 'dims') && isfield(u, 'dparts')
    error('dparts cannot be specified without dims');
end

if isfield(u, 'dims')
    dims = u.dims(:)';
    if ~iscell(dims) || isempty(dims) || ~all(cellfun(@isnumeric, dims))
        error('dims must be a cell array of numeric vectors');
    end
    for i = 1 : numel(dims)
        if isempty(dims{i}) || (size(dims{i}, 1) ~= 1)
            error('dims must contain scalars or nonempty row vectors');
        end
    end
    if ~any([dims{:}] == 1)
        error('dims must contain the number 1 at least once');
    end
    if any(([dims{:}] ~= 1) & ([dims{:}] ~= 2))
        error('dims may only contain the numbers 1 and 2');
    end
else
    dims = {1 2};
end

if isfield(u, 'dparts')
    dparts = u.dparts(:)';
    if ~iscell(dparts) || ~all(cellfun(@isnumeric, dparts))
        error('dparts must be a cell array of numeric vectors');
    end
    if numel(dparts) ~= numel(dims)
        error('dparts must have the same number of elements (%u) as dims', numel(dims));
    end
    for i = 1 : numel(dims)
        if ~isequal(size(dparts{i}), size(dims{i}))
            error('each element of dparts must be the same size as the corresponding element of dims');
        end
    end
    tdims  = [dims{:}  ];
    tparts = [dparts{:}];
    for i = 1 : max(tdims)
        if ~isequal(sort(tparts(tdims == i)), 1 : sum(tdims == i))
            error('parts for dimension %u must uniquely cover the range 1 to %u', i, sum(tdims == i));
        end
    end
else
    dparts = {};
    for i = 1 : numel(dims)
        for j = 1 : numel(dims{i})
            dparts{i}(1, j) = sum([dims{1 : i - 1}, dims{i}(1 : j - 1)] == dims{i}(j)) + 1;
        end
    end
end

if isfield(u, 'dnames')
    dnames = u.dnames(:)';
    if ~iscellstr(dnames)
        error('dnames must be a cell array of strings');
    end
    if numel(dnames) ~= numel(dims)
        error('dnames must have the same number of elements (%u) as dims', numel(dims));
    end
    if any(isstrprop([dnames{:}], 'digit'))
        error('dnames cannot contain digits');
    end
    tnames = dnames(~strcmp(dnames, ''));
    if numel(unique(tnames)) ~= numel(tnames)
        error('dnames must be unique');
    end
elseif isequal(dims, {1 2})
    dnames = {'y' 'x'};
else
    dnames = repmat({''}, 1, numel(dims));
end

if isfield(u, 'dmap')
    dmap = u.dmap(:)';
    if ~islogical(dmap) && ~isnumeric(dmap)
        error('dmap must be a logical or numeric vector');
    end
    dmap = double(dmap);
    if any(~ismember(dmap, [0 1 2])) || (~shiftOK && any(dmap == 2))
        error('invalid dmap value');
    end
    if numel(dmap) ~= numel(dims)
        error('dmap must have the same number of elements (%u) as dims', numel(dims));
    end
    if any(dmap & cellfun(@isempty, dnames))
        error('dmap cannot be true if dnames is null');
    end
    if sum(dmap == 2) > 1
        error('can only have one shiftable dimension');
    end
    if any(dmap == 2)
        if ~isscalar(dims{dmap == 2})
            error('shiftable dimension cannot have multiple parts');
        end
        if (dims{dmap == 2} ~= 2) || (dparts{dmap == 2} ~= sum([dims{:}] == 2))
            error('shiftable dimension must be the last part of internal dimension 2');
        end
    end
else
    dmap = zeros(1, numel(dims));
end

d.dims   = dims;
d.dparts = dparts;
d.dnames = dnames;
d.dmap   = dmap;

return;

%***********************************************************************************************************************

function d = Undef(d)

d.dims   = {};
d.dparts = {};
d.dnames = {};
d.dmap   = [];

return;

%***********************************************************************************************************************

function inds = Iter(d, dnames)

inds  = [];
found = 0;
test{1} = false(0);
test{2} = false(0);

k = 0;
for i = 1 : numel(d.dims)
    if (d.dmap(i) == 2) || isempty(d.dnames{i})
        iter = false;
    else
        iter = any(strcmpi(d.dnames{i}, dnames));
        found = found + iter;
    end
    for j = 1 : numel(d.dims{i})
        k = k + 1;
        if iter, inds(end + 1) = k; end
        test{d.dims{i}(j)}(d.dparts{i}(j)) = iter;
    end
end

if found ~= numel(dnames), error('invalid iteration dimension'); end

for i = 1 : 2
    % You can only iterate over leading and trailing dimensions.
    while ~isempty(test{i}) && test{i}(1  ), test{i}(1  ) = []; end
    while ~isempty(test{i}) && test{i}(end), test{i}(end) = []; end
    if any(test{i}), error('invalid iteration dimension'); end
end

return;