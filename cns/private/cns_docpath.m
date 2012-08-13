function cns_docpath(varargin)

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

cur = cd;

try
    docpath(varargin{:});
catch
end

cd(cur);

return;

%***********************************************************************************************************************

function docpath(userPath, cnsPath)

up = fullfile(userPath, 'doc');
cp = fullfile(cnsPath , 'doc');
if ~exist(up, 'dir'), return; end
if ~exist(cp, 'dir'), return; end

switch cns_osname
case 'linux', fn = 'cnspathlin';
case 'win'  , fn = 'cnspathwin';
case 'mac'  , fn = 'cnspathmac';
otherwise   , return;
end

fh = fopen(fullfile(up, [fn '.js']), 'w');
if fh < 0, return; end

fprintf(fh, 'function %s() {\n', fn);
fprintf(fh, 'return "%s";\n', relpath(up, cp));
fprintf(fh, '}\n');

fclose(fh);

return;

%***********************************************************************************************************************

function rel = relpath(src, dst)

% Get absolute paths without symlinks.
cur = cd(src); src = cd(cur);
cur = cd(dst); dst = cd(cur);

sp = pathparts(src);
dp = pathparts(dst);

i = 1;
while (i <= min(numel(sp), numel(dp))) && strcmp(sp{i}, dp{i})
    i = i + 1;
end

if i <= 2
    rel = dst;
else
    sp = sp(i : end);
    dp = dp(i : end);
    if isempty(sp) && isempty(dp)
        rel = '.';
    else
        rel = [repmat({'..'}, 1, numel(sp)), dp];
        if numel(rel) == 1
            rel = rel{1};
        else
            rel = fullfile(rel{:});
        end
    end
end

rel(rel == filesep) = '/';

return;

%***********************************************************************************************************************

function parts = pathparts(path)

parts = {};

while true

    [path, name, ext] = fileparts(path);
    name = [name ext];
    if isempty(name), break; end

    parts = [{name} parts];

end

parts = [{path} parts];

return;