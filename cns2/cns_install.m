function cns_install

% CNS_INSTALL
%    Click <a href="matlab: cns_help('cns_install')">here</a> for help.

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

Compile('private', 'cns_initsynapses');
Compile('private', 'cns_intin');
Compile('private', 'cns_intout');
Compile('private', 'cns_limits');
Compile('private', 'cns_sync', true, sprintf('-D_SYNC_INC=sync_%s_inc.h', cns_osname));

Build('private', 'cns_devcount_cuda', 'cuda', 'cu');

Compile(fullfile('util', 'private'), 'cns_spikeutil');

rehash path;

return;

%***********************************************************************************************************************

function Compile(outdn, func, quiet, varargin)

if nargin < 3, quiet = false; end

base = fileparts(mfilename('fullpath'));

outdp = fullfile(base, outdn);
outfp = fullfile(outdp, [func '.' mexext]);
srcfp = fullfile(base, 'source', [func '.cpp']);

if ~exist(outdp, 'dir'), mkdir(outdp); end
if exist(outfp, 'file'), delete(outfp); end

if quiet
    [ans, ans] = system(['mex', sprintf(' %s', varargin{:}, '-outdir', outdp, srcfp)]);
else
    mex(varargin{:}, '-outdir', outdp, srcfp);
end

return;

%***********************************************************************************************************************

function Build(outdn, func, platform, srcext)

base = fileparts(mfilename('fullpath'));

outdp  = fullfile(base, outdn);
outfp  = fullfile(outdp, [func '.' mexext]);
srcdp  = fullfile(base, 'source');
srcfp  = fullfile(srcdp, [func '.' srcext]);
tmpfp1 = fullfile(srcdp, [func '.cpp']);
tmpfp2 = fullfile(srcdp, [func '.' mexext]);
tmpfp3 = fullfile(srcdp, [func '.linkinfo']); % spurious file to be deleted

if ~exist(outdp, 'dir'), mkdir(outdp); end
if exist(outfp, 'file'), delete(outfp); end

if exist(tmpfp1, 'file'), delete(tmpfp1); end
if exist(tmpfp2, 'file'), delete(tmpfp2); end

path = cd(srcdp);
[status, output] = cns_compile(platform, 'compile', srcfp, tmpfp1, tmpfp2);
cd(path);

if exist(tmpfp1, 'file'), delete(tmpfp1); end
if exist(tmpfp2, 'file'), movefile(tmpfp2, outfp); end
if exist(tmpfp3, 'file'), delete(tmpfp3); end

if status == 1
    fprintf(output);
end

return;