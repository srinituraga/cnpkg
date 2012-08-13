function [val, p] = cns_prepimage(im, bufSize, nSpace)

% CNS_PREPIMAGE
%    Click <a href="matlab: cns_help('cns_prepimage')">here</a> for help.

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

if ischar(im), im = imread(im); end
if size(im, 3) == 3, im = rgb2gray(im); end
if isinteger(im)
    im = single(im) / single(intmax(class(im)));
else
    im = single(im);
end

siz = size(im);
if any(siz > bufSize)
    siz = round(siz * min(bufSize ./ siz));
    im = imresize(im, siz);
end

if all(siz == bufSize)
    val = im;
else
    val = zeros(bufSize, 'single');
    val(1 : siz(1), 1 : siz(2)) = im;
end

coverage = (siz .* nSpace) / max(siz .* nSpace);
space = coverage ./ siz;
start = 0.5 - 0.5 * (siz - 1) .* space;

% The following adjustment would make the results more like those of imresize.
% start = start + [0.0 -0.5] .* space;

p.size  = siz;
p.start = start;
p.space = space;

return;