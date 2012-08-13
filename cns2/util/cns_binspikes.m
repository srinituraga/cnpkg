function bins = cns_binspikes(times, dur, dt, bdt)

% CNS_BINSPIKES
%    Click <a href="matlab: cns_help('cns_binspikes')">here</a> for help.

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

if nargin < 4, bdt = dt; end

siz = size(times);

bins = cns_spikeutil(2, int32(times), round(dur / bdt), round(bdt / dt));

bins = reshape(bins, [size(bins, 1), siz(2 : end)]);

return;