function cns_rename(oname, nname, varargin)

% CNS_RENAME
%    Click <a href="matlab: cns_help('cns_rename')">here</a> for help.

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

subdirs = varargin;

if strcmp(oname, nname), return; end
if ~isvarname(nname) || (nname(end) == '_')
    error('"%s" is not a valid package name', nname);
end

base = which([oname '_cns'], '-all');
if numel(base) == 0
    error('"%s_cns.m" is not in the MATLAB path', oname);
end
if numel(base) > 1
    error('"%s_cns.m" is in the MATLAB path in more than one place', oname);
end
base = fileparts(base{1});

if numel(which([nname '_cns'], '-all')) > 0
    error('"%s_cns.m" is already in the MATLAB path', nname);
end

cns_build(oname, 'clean');

ofiles = dir(fullfile(base, [oname '_*']));
ofiles = {ofiles(~[ofiles.isdir]).name};
paths  = repmat({base}, 1, numel(ofiles));
for i = 1 : numel(subdirs)
    files = dir(fullfile(base, subdirs{i}, [oname '_*']));
    files = {files(~[files.isdir]).name};
    ofiles(end + 1 : end + numel(files)) = files;
    paths (end + 1 : end + numel(files)) = {fullfile(base, subdirs{i})};
end

nfiles = cell(1, numel(ofiles));
for i = 1 : numel(ofiles)
    nfiles{i} = [nname '_' ofiles{i}(numel(oname) + 2 : end)];
    if exist(fullfile(paths{i}, nfiles{i}), 'file')
        error('"%s" already exists', nfiles{i});
    end
end

for i = 1 : numel(ofiles)
    if strcmp(ofiles{i}(end - 1 : end), '.m')
        CopyFunc(paths{i}, ofiles{i}(1 : end - 2), nfiles{i}(1 : end - 2));
        delete(fullfile(paths{i}, ofiles{i}));
    else
        if ~movefile(fullfile(paths{i}, ofiles{i}), fullfile(paths{i}, nfiles{i}))
            error('error renaming "%s" to "%s"', ofiles{i}, nfiles{i});
        end
    end
end

return;

%***********************************************************************************************************************

function CopyFunc(path, oname, nname)

oh = fopen(fullfile(path, [oname '.m']), 'r');
if oh < 0
    error('unable to read "%s.m"', oname);
end

nh = fopen(fullfile(path, [nname '.m']), 'w');
if nh < 0
    fclose(oh);
    error('unable to write "%s.m"', nname);
end

try
    CopyFunc2(oh, nh, oname, nname);
catch
    err = lasterror;
    fclose(oh);
    fclose(nh);
    rethrow(err);
end

fclose(oh);
fclose(nh);

return;

%***********************************************************************************************************************

function CopyFunc2(oh, nh, oname, nname)

exps = {};
exps{end + 1} = '^function\s+.+\s*=\s*(.+)\(';
exps{end + 1} = '^function\s+.+\s*=\s*(.+)$';
exps{end + 1} = '^function\s+(.+)\(';
exps{end + 1} = '^function\s+(.+)$';

found = false;

while true

    line = fgetl(oh);
    if ~ischar(line), break; end

    if ~found && strcmp(strtok(line), 'function')
        for i = 1 : numel(exps)
            p = regexp(line, exps{i}, 'tokenExtents', 'once');
            if ~isempty(p), break; end
        end
        if isempty(p)
            error('function name not found in "%s.m"', oname);
        end
        line = [line(1 : p(1) - 1), nname, line(p(2) + 1 : end)];
        found = true;
    end

    fprintf(nh, '%s\n', line);

end

return;