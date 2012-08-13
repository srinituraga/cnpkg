function cns_help(varargin)

% CNS
%    Click <a href="matlab: cns_help('index')">here</a> for help.

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

switch nargin
case 0   , package = ''         ; page = 'index';
case 1   , package = ''         ; page = varargin{1};
case 2   , package = varargin{1}; page = varargin{2};
otherwise, error('too many arguments');
end

i = find(page == '#', 1);
if isempty(i)
    part = '';
else
    part = page(i + 1 : end);
    page = page(1 : i - 1);
end

if isempty(package)

    fn = fullfile(fileparts(mfilename('fullpath')), 'doc', [page '.html']);

    if ~exist(fn, 'file')
        fprintf('Sorry, help for "%s" is not yet written.\n', page);
        fprintf('Opening main CNS help page....\n');
        fn   = fullfile(fileparts(mfilename('fullpath')), 'doc', 'index.html');
        part = '';
    end

else

    path = fileparts(which([package '_cns']));
    if isempty(path), error('cannot locate "%s_cns.m"', package); end
    fn = fullfile(path, 'doc', [page '.html']);

    if ~exist(fn, 'file')
        fprintf('Sorry, help for "%s" is not yet written.\n', page);
        return;
    end

end

if isempty(part)
    url = ['file://' fn];
else
    url = ['file://' fn '#' part];
end

web(url, '-browser');

return;