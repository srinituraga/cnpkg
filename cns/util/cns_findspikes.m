function times = cns_findspikes(v, thres, dt, vdt)

% CNS_FINDSPIKES
%    Click <a href="matlab: cns_help('cns_findspikes')">here</a> for help.

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

if nargin < 2, thres = 0 ; end
if nargin < 3, dt    = 1 ; end
if nargin < 4, vdt   = dt; end

siz = size(v);
if siz(1) < 2, error('not enough samples'); end

times = cns_spikeutil(1, single(v), thres, round(vdt / dt));

times = reshape(times, [size(times, 1), siz(2 : end)]);

return;