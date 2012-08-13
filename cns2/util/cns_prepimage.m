function [p, val] = cns_prepimage(bufSize, n, im, method, varargin)

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

if nargin < 4, method = 'fit'; end

if isnumeric(im) && (numel(im) == 2)
    p.size = im(:)';
    im = [];
else
    if ischar(im), im = imread(im); end
    if size(im, 3) == 3, im = rgb2gray(im); end
    if isinteger(im)
        im = single(im) / single(intmax(class(im)));
    else
        im = single(im);
    end
    p.size = size(im);
end

switch method
case 'fit'   , [p, im] = Fit   (bufSize, n, p, im, varargin{:});
case 'center', [p, im] = Center(bufSize, n, p, im, varargin{:});
otherwise    , error('invalid method');
end

% The following adjustment would make the results more like those of imresize.
% p.start = p.start + [0.0 -0.5] .* p.space;

if isempty(im)
    val = [];
elseif all(p.size == bufSize)
    val = im;
else
    val = zeros(bufSize, 'single');
    val(1 : p.size(1), 1 : p.size(2)) = im;
end

return;

%***********************************************************************************************************************

function [p, im] = Fit(bufSize, n, p, im)

if any(p.size > bufSize)
    if isempty(im), error('image too large'); end
    p.size = round(p.size * min(bufSize ./ p.size));
    im = imresize(im, p.size);
end

p.space = n.space * min(n.size ./ p.size);
p.start = n.start + 0.5 * (n.size - 1) .* n.space - 0.5 * (p.size - 1) .* p.space;

return;

%***********************************************************************************************************************

function [p, im] = Center(bufSize, n, p, im, factor)

if any(p.size > bufSize), error('image too large'); end

p.space = n.space * factor;
p.start = n.start + 0.5 * (n.size - 1) .* n.space - 0.5 * (p.size - 1) .* p.space;

return;