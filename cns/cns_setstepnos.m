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

function m = Field(m, fields)

if ischar(fields), fields = {fields}; end

def = cns_def(m);

p = repmat(0 , 0, numel(def.layers));
r = repmat(-1, 1, numel(def.layers));

for z = 1 : numel(def.layers)

    a = [];
    for i = 1 : numel(fields)
        a = [a, m.layers{z}.(fields{i})];
    end
    a = unique(a);
    if any(a(:) < 0                ), error('z=%u: invalid previous layer', z); end
    if any(a(:) > numel(def.layers)), error('z=%u: invalid previous layer', z); end
    a = a(a > 0);
    if ~def.layers{z}.kernel && ~isempty(a)
        error('z=%u: layers with no kernel cannot have dependencies', z);
    end

    if isfield(m.layers{z}, 'stepSkip') && m.layers{z}.stepSkip
        a = [];
    end

    p(1 : numel(a), z) = a(:);
    if isempty(a), r(z) = 0; end

end

p(ismember(p, find(r == 0))) = 0;
n = 1;

while true

    f = find((r < 0) & all(p == 0, 1));
    if isempty(f), break; end

    r(f) = n;
    n = n + 1;

    p(ismember(p, f)) = 0;

end

if any(r < 0), error('there is a dependency loop'); end

for z = 1 : numel(def.layers)
    if r(z) == 0
        m.layers{z}.stepNo = [];
    else
        m.layers{z}.stepNo = r(z);
    end
end

return;
