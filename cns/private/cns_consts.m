function c = cns_consts(mode, varargin)

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

switch mode
case 'layertable', c = LayerTable(varargin{:});
otherwise        , error('invalid mode');
end

return;

%***********************************************************************************************************************

function c = LayerTable(maxSiz2p)

c.len    = ceil((10 + maxSiz2p) / 2) * 2;
c.gmvOff = 0;
c.mvOff  = 1;
c.gcOff  = 2;
c.cOff   = 3;
c.tOff   = 4;
c.xCount = 5;
c.nwOff  = 6; % 4 bytes
c.ndOff  = 8; % 4 bytes
c.siz2p  = 10;

return;