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

base = which(oname, '-all');
if numel(base) == 0
    error('"%s.m" is not in the MATLAB path', oname);
end
if numel(base) > 1
    error('"%s.m" is in the MATLAB path in more than one place', oname);
end
base = fileparts(base{1});

if numel(which(nname, '-all')) > 0
    error('"%s.m" is already in the MATLAB path', nname);
end

cns_build(oname, 'clean');

ofiles = dir(fullfile(base, [oname '.*']));
ofiles = [ofiles; dir(fullfile(base, [oname '_*']))];
ofiles = {ofiles(~[ofiles.isdir]).name};
paths  = repmat({base}, 1, numel(ofiles));
for i = 1 : numel(subdirs)
    files = dir(fullfile(base, subdirs{i}, [oname '.*']));
    files = [files; dir(fullfile(base, subdirs{i}, [oname '_*']))];
    files = {files(~[files.isdir]).name};
    ofiles(end + 1 : end + numel(files)) = files;
    paths (end + 1 : end + numel(files)) = {fullfile(base, subdirs{i})};
end

nfiles = cell(1, numel(ofiles));
for i = 1 : numel(ofiles)
    nfiles{i} = [nname, ofiles{i}(numel(oname) + 1 : end)];
    if exist(fullfile(paths{i}, nfiles{i}), 'file')
        error('"%s" already exists', nfiles{i});
    end
end

for real = [false true]
    for i = 1 : numel(ofiles)
        if strcmp(ofiles{i}(end - 1 : end), '.m')
            CopyMFile(real, paths{i}, ofiles{i}(1 : end - 2), nfiles{i}(1 : end - 2));
            if real, delete(fullfile(paths{i}, ofiles{i})); end
        elseif real
            if ~movefile(fullfile(paths{i}, ofiles{i}), fullfile(paths{i}, nfiles{i}))
                error('error renaming "%s" to "%s"', ofiles{i}, nfiles{i});
            end
        end
    end
end

return;

%***********************************************************************************************************************

function CopyMFile(real, path, oname, nname)

oh = fopen(fullfile(path, [oname '.m']), 'r');
if oh < 0
    error('unable to read "%s.m"', oname);
end

if real
    nh = fopen(fullfile(path, [nname '.m']), 'w');
    if nh < 0
        fclose(oh);
        error('unable to write "%s.m"', nname);
    end
else
    nh = [];
end

try
    CopyMFile2(oh, nh, oname, nname);
catch
    err = lasterror;
    fclose(oh);
    if ~isempty(nh), fclose(nh); end
    rethrow(err);
end

fclose(oh);
if ~isempty(nh), fclose(nh); end

return;

%***********************************************************************************************************************

function CopyMFile2(oh, nh, oname, nname)

first = true;

while true

    line = fgetl(oh);
    if ~ischar(line), break; end

    if first
        tok = strtok(line);
        if isempty(tok) || (tok(1) == '%')
        elseif strcmp(tok, 'classdef')
            line = FixClass(deblank(line), oname, nname);
            first = false;
        elseif strcmp(tok, 'function')
            line = FixFunc(deblank(line), oname, nname);
            first = false;
        else
            first = false;
        end
    end

    if ~isempty(nh), fprintf(nh, '%s\n', line); end

end

return;

%***********************************************************************************************************************

function line = FixClass(line, oname, nname)

% TODO

return;

%***********************************************************************************************************************

function line = FixFunc(line, oname, nname)

exps = {};
exps{end + 1} = '^function\s+.+\s*=\s*(.+)\(';
exps{end + 1} = '^function\s+.+\s*=\s*(.+)$';
exps{end + 1} = '^function\s+(.+)\(';
exps{end + 1} = '^function\s+(.+)$';

for i = 1 : numel(exps)
    p = regexp(line, exps{i}, 'tokenExtents', 'once');
    if ~isempty(p), break; end
end
if isempty(p)
    error('unable to parse "function" line in "%s.m"', oname);
end

line = [line(1 : p(1) - 1), nname, line(p(2) + 1 : end)];

return;