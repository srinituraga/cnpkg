function m = cns_mapdim(m, z, dimID, method, varargin)

% CNS_MAPDIM
%    Click <a href="matlab: cns_help('cns_mapdim')">here</a> for help.

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

switch method
case 'copy'        , m = MapCopy        (m, z, dimID, varargin{:});
case 'pixels'      , m = MapPixels      (m, z, dimID, varargin{:});
case 'scaledpixels', m = MapScaledPixels(m, z, dimID, varargin{:});
case 'int'         , m = MapInt         (m, z, dimID, varargin{:});
case 'int-td'      , m = MapIntTD       (m, z, dimID, varargin{:});
case 'win'         , m = MapWin         (m, z, dimID, varargin{:});
case 'temp1'       , m = MapTemp1       (m, z, dimID, varargin{:});
case 'temp2'       , m = MapTemp2       (m, z, dimID, varargin{:});
otherwise          , error('invalid method');
end

return;

%***********************************************************************************************************************

function m = MapCopy(m, z, dimID, pz)

[d2, name, map2] = cns_finddim(m, z, dimID);

d1 = cns_finddim(m, pz, name, map2 : 2);

m.layers{z}.size{d2} = m.layers{pz}.size{d1};

if map2 >= 1
    m.layers{z}.([name '_start']) = m.layers{pz}.([name '_start']);
    m.layers{z}.([name '_space']) = m.layers{pz}.([name '_space']);
end

if map2 >= 2
    m.layers{z}.([name '_lag'  ]) = m.layers{pz}.([name '_lag'  ]);
    m.layers{z}.([name '_total']) = m.layers{pz}.([name '_total']);
    m.layers{z}.([name '_shift']) = m.layers{pz}.([name '_shift']);
end

return;

%***********************************************************************************************************************

function m = MapPixels(m, z, dimID, imSize, range)

if nargin < 5, range = [0 1]; end

[d2, name] = cns_finddim(m, z, dimID, 1);

if (imSize < 1) || (mod(imSize, 1) ~= 0)
    error('imSize must be a positive integer');
end
if (numel(range) ~= 2) || (diff(range) <= 0)
    error('invalid range');
end

nSpace = diff(range) / imSize;

nStart = mean(range) - 0.5 * nSpace * (imSize - 1);

m.layers{z}.size{d2}          = imSize;
m.layers{z}.([name '_start']) = nStart;
m.layers{z}.([name '_space']) = nSpace;

return;

%***********************************************************************************************************************

function m = MapScaledPixels(m, z, dimID, baseSize, factor, range)

if nargin < 6, range = [0 1]; end

[d2, name] = cns_finddim(m, z, dimID, 1);

if (baseSize < 1) || (mod(baseSize, 1) ~= 0)
    error('baseSize must be a positive integer');
end
if factor <= 0
    error('factor must be positive');
end
if (numel(range) ~= 2) || (diff(range) <= 0)
    error('invalid range');
end

nSpace = diff(range) / baseSize * factor;

if mod(baseSize, 2) == 1
    nSize = max(0, 2 * floor((diff(range) - nSpace) / (2 * nSpace)) + 1);
else
    nSize = max(0, 2 * floor(diff(range) / (2 * nSpace)));
end

nStart = mean(range) - 0.5 * nSpace * (nSize - 1);

m.layers{z}.size{d2}          = nSize;
m.layers{z}.([name '_start']) = nStart;
m.layers{z}.([name '_space']) = nSpace;

return;

%***********************************************************************************************************************

function m = MapInt(m, z, dimID, pz, rfSize, rfStep, parity)

% rfSize = unit width in previous layer units.
% rfStep = unit step size in previous layer units.
% parity = 0 if you prefer an even-sized output, 1 if odd, [] if you don't care.

if nargin < 7, parity = []; end

[d2, name] = cns_finddim(m, z, dimID, 1);

d1 = cns_finddim(m, pz, name, 1);

pSize  = m.layers{pz}.size{d1};
pStart = m.layers{pz}.([name '_start']);
pSpace = m.layers{pz}.([name '_space']);

pCenter = pStart + 0.5 * pSpace * (pSize - 1);

if rfSize >= cns_intmax

    m.layers{z}.size{d2}          = 1;
    m.layers{z}.([name '_start']) = pCenter;
    m.layers{z}.([name '_space']) = 1;

    return;

end

if (rfSize < 1) || (mod(rfSize, 1) ~= 0)
    error('rfSize must be a positive integer');
end
if (rfStep < 1) || (mod(rfStep, 1) ~= 0)
    error('rfStep must be a positive integer');
end
if ~isempty(parity) && ~any(parity == [0 1])
    error('parity must be 0, 1, or empty');
end

nSpace = pSpace * rfStep;

nSize1 = max(0, 2 * floor((pSize - rfSize         ) / (2 * rfStep)) + 1);
nSize0 = max(0, 2 * floor((pSize - rfSize - rfStep) / (2 * rfStep)) + 2);

if mod(pSize, 2) == mod(rfSize, 2)

    if mod(rfStep, 2) == 0

        % We can place a unit in the center, or not.

        if isequal(parity, 1) || (isempty(parity) && (nSize1 >= nSize0))
            nSize = nSize1;
        else
            nSize = nSize0;
        end

    else

        % We must place a unit in the center.  The result will have an odd number of units.

        nSize = nSize1;

    end

else

    % We cannot place a unit in the center, so the result will have an even number of units, and we must place a unit
    % on either side of the center, at the same distance from the center.  This is only possible if rfStep is odd.
    % This really requires a diagram to see.  There are two cases to consider: pSize odd, rfSize even and vice-versa.

    nSize = nSize0;

    if (nSize > 0) && (mod(rfStep, 2) == 0)
        error('when the result layer has an even number of units, rfStep must be odd');
    end

end

nStart = pCenter - 0.5 * nSpace * (nSize - 1);

m.layers{z}.size{d2}          = nSize;
m.layers{z}.([name '_start']) = nStart;
m.layers{z}.([name '_space']) = nSpace;

return;

%***********************************************************************************************************************

function m = MapIntTD(m, z, dimID, pz, rfSize, rfStep)

[d2, name] = cns_finddim(m, z, dimID, 1);

d1 = cns_finddim(m, pz, name, 1);

pSize  = m.layers{pz}.size{d1};
pStart = m.layers{pz}.([name '_start']);
pSpace = m.layers{pz}.([name '_space']);

pCenter = pStart + 0.5 * pSpace * (pSize - 1);

if (rfSize < 1) || (mod(rfSize, 1) ~= 0)
    error('rfSize must be a positive integer');
end
if (rfStep < 1) || (mod(rfStep, 1) ~= 0)
    error('rfStep must be a positive integer');
end

nSpace = pSpace / rfStep;

if pSize == 0
    nSize = 0;
else
    nSize = rfSize + rfStep * (pSize - 1);
end

nStart = pCenter - 0.5 * nSpace * (nSize - 1);

m.layers{z}.size{d2}          = nSize;
m.layers{z}.([name '_start']) = nStart;
m.layers{z}.([name '_space']) = nSpace;

return;

%***********************************************************************************************************************

function m = MapWin(m, z, dimID, pz, rfSize, rfStep, rfMargin, parity)

% rfSize   = window width in previous layer units (can be fractional).
% rfStep   = window step size in previous layer units (can be fractional).
% rfMargin = size of margin in previous layer units (can be fractional and/or negative).
% parity   = 0 if you prefer an even-sized output, 1 if odd, [] if you don't care.

if nargin < 8, parity = []; end

[d2, name] = cns_finddim(m, z, dimID, 1);

d1 = cns_finddim(m, pz, name, 1);

pSize  = m.layers{pz}.size{d1};
pStart = m.layers{pz}.([name '_start']);
pSpace = m.layers{pz}.([name '_space']);

pCenter = pStart + 0.5 * pSpace * (pSize - 1);
pWidth  = pSize * pSpace;

if rfSize >= cns_intmax

    m.layers{z}.size{d2}          = 1;
    m.layers{z}.([name '_start']) = pCenter;
    m.layers{z}.([name '_space']) = 1;

    return;

end

if rfSize <= 0
    error('rfSize must be positive');
end
if rfStep <= 0
    error('rfStep must be positive');
end
if ~isempty(parity) && ~any(parity == [0 1])
    error('parity must be 0, 1, or empty');
end

rfSize   = rfSize   * pSpace;
rfStep   = rfStep   * pSpace;
rfMargin = rfMargin * pSpace;

nSize1 = max(0, 2 * floor((pWidth - 2 * rfMargin - rfSize         ) / (2 * rfStep)) + 1);
nSize0 = max(0, 2 * floor((pWidth - 2 * rfMargin - rfSize - rfStep) / (2 * rfStep)) + 2);

if isequal(parity, 1) || (isempty(parity) && (nSize1 >= nSize0))
    nSize = nSize1;
else
    nSize = nSize0;
end

nStart = pCenter - 0.5 * rfStep * (nSize - 1);

m.layers{z}.size{d2}          = nSize;
m.layers{z}.([name '_start']) = nStart;
m.layers{z}.([name '_space']) = rfStep;

return;

%***********************************************************************************************************************

function m = MapTemp1(m, z, dimID, maxShift)

[d2, name] = cns_finddim(m, z, dimID, 2);

if (maxShift < 1) || (mod(maxShift, 1) ~= 0)
    error('maxShift must be a positive integer');
end

if isfield(m, [name '_maxShift'])
    error('maxShift is already defined');
end

m.([name '_maxShift']) = maxShift;
m.([name '_current' ]) = 0;

m.layers{z}.size{d2}          = maxShift;
m.layers{z}.([name '_start']) = 0.5 - maxShift;
m.layers{z}.([name '_space']) = 1;
m.layers{z}.([name '_lag'  ]) = 0.5;
m.layers{z}.([name '_total']) = 0;
m.layers{z}.([name '_shift']) = 0;

return;

%***********************************************************************************************************************

function m = MapTemp2(m, z, dimID, pzs, rfSizes, rfStep)

% rfSize = unit width in previous layer units.
% rfStep = unit step size in previous layer units.

[d2, name] = cns_finddim(m, z, dimID, 2);

if any(rfSizes < 1) || any(mod(rfSizes, 1) ~= 0)
    error('rfSize must be a positive integer');
end
if (rfStep < 1) || (mod(rfStep, 1) ~= 0)
    error('rfStep must be a positive integer');
end

maxShift = m.([name '_maxShift']);

pSpace = m.layers{pzs(1)}.([name '_space']);
pLag   = m.layers{pzs(1)}.([name '_lag'  ]);

nSpace = pSpace * rfStep;
nLag = pLag + pSpace * 0.5 * (rfSizes(1) - 1);
nSize = ceil(maxShift / nSpace);
nStart = nLag - nSpace * nSize;

m.layers{z}.size{d2}          = nSize;
m.layers{z}.([name '_start']) = nStart;
m.layers{z}.([name '_space']) = nSpace;
m.layers{z}.([name '_lag'  ]) = nLag;
m.layers{z}.([name '_total']) = 0;
m.layers{z}.([name '_shift']) = 0;

% Now see if any layers we depend on need to be bigger.

for i = 1 : numel(pzs)

    d1 = cns_finddim(m, pzs(i), name, 2);

    pSize  = m.layers{pzs(i)}.size{d1};
    pStart = m.layers{pzs(i)}.([name '_start']);
    pSpace = m.layers{pzs(i)}.([name '_space']);
    pLag   = m.layers{pzs(i)}.([name '_lag'  ]);

    if mod(nSpace, pSpace) ~= 0
        error('layer %s_space values must be integer multiples', name);
    end

    % Find the time just before the new layer's first shift, plus one complete cycle.
    % Shift both layers to those times.
    times = nLag + nStart + nSize * nSpace + (-1 : nSpace - 2);
    nStart2 = nStart + nSpace * GetShift(nSize, nStart, nSpace, nLag, times);
    pStart2 = pStart + pSpace * GetShift(pSize, pStart, pSpace, pLag, times);

    % Now perform the maximum shift.
    % Find the earliest frame in the previous layer (needed and actual).
    times = times + maxShift;
    needed = nStart2 + nSpace * nSize - 0.5 * (rfSizes(i) - 1) * pSpace;
    actual = pStart2 + pSpace * GetShift(pSize, pStart2, pSpace, pLag, times);

    % See how many extra frames we need in the previous layer.
    extra = round((actual - needed) / pSpace);
    pSizeMin = pSize + max([extra 0]);

    if pSize < pSizeMin
        m.layers{pzs(i)}.size{d1}          = pSizeMin;
        m.layers{pzs(i)}.([name '_start']) = pLag - pSpace * pSizeMin;
    end

end

return;

%-----------------------------------------------------------------------------------------------------------------------

function shift = GetShift(siz, start, space, lag, time)

% Formula copied from cns('shift').
shift = max(1 + floor((time - lag - start) ./ space + 0.001) - siz, 0);

return;