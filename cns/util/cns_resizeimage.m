function out = cns_resizeimage(im, oSize, fill)

% CNS_RESIZEIMAGE
%    Click <a href="matlab: cns_help('cns_resizeimage')">here</a> for help.

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

if nargin < 3, fill = 0; end

iSize = [size(im, 1), size(im, 2)];
rSize = round(iSize * min(oSize ./ iSize));
off = floor(0.5 * (oSize - rSize));

out = repmat(feval(class(im), fill), [oSize, size(im, 3)]);
out(off(1) + 1 : off(1) + rSize(1), off(2) + 1 : off(2) + rSize(2), :) = imresize(im, rSize);

return;