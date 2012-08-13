function v = cns_version

% Internal CNS function.

%***********************************************************************************************************************

% Copyright (C) 2010 by Jim Mutch (www.jimmutch.com).
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

v = 0;

stack = dbstack('-completenames');
if numel(stack) < 2, return; end
path = fullfile(fileparts(stack(2).file), 'Version.txt');

fid = fopen(path, 'r');
if fid < 0, return; end
line = fgetl(fid);
fclose(fid);
if isequal(line, -1), return; end

[ans, str] = strtok(line, ':');
if ~isempty(str) && (str(1) == ':'), str = str(2 : end); end
str = strtrim(str);
if isempty(str) || ~all(isstrprop(str, 'digit')), return; end
num = str2double(str);
if ~isfinite(num), return; end

v = num;

return;