function [status, output] = cns_compile(platform, option, inputFilePath, varargin)

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

scriptPath = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'scripts', cns_osname);
if ~exist(scriptPath, 'dir'), error('unsupported operating system'); end

compileCommand = Quoted(fullfile(scriptPath, ['compile_' platform]));
mexIncludePath = Quoted(fullfile(matlabroot, 'extern', 'include'));
setupCommand   = Quoted(fullfile(scriptPath, 'setup'));

cmd = [compileCommand ' ' mexIncludePath ' ' setupCommand ' ' Quoted(inputFilePath) ' ' option];

for i = 1 : numel(varargin)
    cmd = [cmd ' ' Quoted(varargin{i})];
end

[status, output] = system(cmd);

return;

%***********************************************************************************************************************

function filePath = Quoted(filePath)

if any(filePath == ' '), filePath = ['"' filePath '"']; end

return;