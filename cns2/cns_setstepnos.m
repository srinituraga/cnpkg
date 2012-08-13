function m = cns_setstepnos(m, mode, varargin)

% CNS_SETSTEPNOS
%    Click <a href="matlab: cns_help('cns_setstepnos')">here</a> for help.

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
case 'field', m = Field(m, varargin{:});
otherwise   , error('invalid mode');
end

return;

%***********************************************************************************************************************

function m = Field(m, fields, start)

if nargin < 3, start = 0; end

if ischar(fields), fields = {fields}; end

p = repmat(0 , 0, numel(m.layers)); % dependencies

for z = 1 : numel(m.layers)

    a = [];
    for i = 1 : numel(fields)
        a = [a, m.layers{z}.(fields{i})];
    end
    a = unique(a);
    if any(a(:) < 0              ), error('z=%u: invalid previous layer', z); end
    if any(a(:) > numel(m.layers)), error('z=%u: invalid previous layer', z); end
    a = a(a > 0);

    if isfield(m.layers{z}, 'stepSkip') && m.layers{z}.stepSkip
        a = [];
    end

    p(1 : numel(a), z) = a(:);

end

r = repmat(-1, 1, numel(m.layers)); % step number
n = start;

while true

    f = find((r < 0) & all(p == 0, 1));
    if isempty(f), break; end

    p(ismember(p, f)) = 0;

    r(f) = n;
    n = n + 1;

end

if any(r < 0), error('there is a dependency loop'); end

for z = 1 : numel(m.layers)
    if r(z) == 0
        m.layers{z}.stepNo = [];
    else
        m.layers{z}.stepNo = r(z);
    end
end

return;
