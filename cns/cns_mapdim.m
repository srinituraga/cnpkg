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
case 'copy'            , m = MapCopy           (m, z, dimID, varargin{:});
case 'pixels'          , m = MapPixels         (m, z, dimID, varargin{:});
case 'scaledpixels'    , m = MapScaledPixels   (m, z, dimID, varargin{:});
case 'scaledpixels-old', m = MapScaledPixelsOld(m, z, dimID, varargin{:});
case 'int'             , m = MapInt            (m, z, dimID, varargin{:});
case 'int-old'         , m = MapIntOld         (m, z, dimID, varargin{:});
case 'int-td'          , m = MapIntTD          (m, z, dimID, varargin{:});
case 'win'             , m = MapWin            (m, z, dimID, varargin{:});
case 'temp1'           , m = MapTemp1          (m, z, dimID, varargin{:});
case 'temp2'           , m = MapTemp2          (m, z, dimID, varargin{:});
otherwise              , error('invalid method');
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
    m.layers{z}.([name '_lag'     ]) = m.layers{pz}.([name '_lag'     ]);
    m.layers{z}.([name '_maxShift']) = m.layers{pz}.([name '_maxShift']);
    m.layers{z}.([name '_total'   ]) = m.layers{pz}.([name '_total'   ]);
    m.layers{z}.([name '_shift'   ]) = m.layers{pz}.([name '_shift'   ]);
end

return;

%***********************************************************************************************************************

function m = MapPixels(m, z, dimID, imSize)

[d2, name] = cns_finddim(m, z, dimID, 1);

if (imSize < 1) || (mod(imSize, 1) ~= 0)
    error('imSize must be a positive integer');
end

m.layers{z}.size{d2}          = imSize;
m.layers{z}.([name '_start']) = 0.5 / imSize;
m.layers{z}.([name '_space']) = 1 / imSize;

return;

%***********************************************************************************************************************

function m = MapScaledPixels(m, z, dimID, baseSize, factor)

[d2, name] = cns_finddim(m, z, dimID, 1);

if (baseSize < 1) || (mod(baseSize, 1) ~= 0)
    error('baseSize must be a positive integer');
end
if factor < 1
    error('factor must be at least 1');
end

nSpace = factor / baseSize;

if mod(baseSize, 2) == 1
    nSize = 2 * floor((1 - nSpace) / (2 * nSpace)) + 1;
else
    nSize = 2 * floor(1 / (2 * nSpace));
end

if nSize < 1
    nSize  = 0;
    nStart = 0.5;
else
    nStart = 0.5 - nSpace * (nSize - 1) / 2;
end

m.layers{z}.size{d2}          = nSize;
m.layers{z}.([name '_start']) = nStart;
m.layers{z}.([name '_space']) = nSpace;

return;

%***********************************************************************************************************************

function m = MapScaledPixelsOld(m, z, dimID, baseSize, factor)

% This is the way FHLib laid out pixels.

[d2, name] = cns_finddim(m, z, dimID, 1);

if (baseSize < 1) || (mod(baseSize, 1) ~= 0)
    error('baseSize must be a positive integer');
end
if factor < 1
    error('factor must be at least 1');
end

nSize = round(baseSize / factor);

if nSize < 1
    nSize  = 0;
    nStart = 0.5;
else
    nStart = 0.5 / nSize;
end

nSpace = factor / baseSize;

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

if rfSize >= cns_intmax

    m.layers{z}.size{d2}          = 1;
    m.layers{z}.([name '_start']) = 0.5;
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

pSize  = m.layers{pz}.size{d1};
pStart = m.layers{pz}.([name '_start']);
pSpace = m.layers{pz}.([name '_space']);

nSpace = pSpace * rfStep;

nSize1 = 2 * floor((pSize - rfSize         ) / (2 * rfStep)) + 1;
nSize0 = 2 * floor((pSize - rfSize - rfStep) / (2 * rfStep)) + 2;

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

    if mod(rfStep, 2) == 0
        error('when the result layer has an even number of units, rfStep must be odd');
    end

    nSize = nSize0;

end

if nSize < 1
    nSize  = 0;
    nStart = 0.5;
else
    nStart = pStart + (pSpace * (pSize - 1) - nSpace * (nSize - 1)) / 2;
end

m.layers{z}.size{d2}          = nSize;
m.layers{z}.([name '_start']) = nStart;
m.layers{z}.([name '_space']) = nSpace;

return;

%***********************************************************************************************************************

function m = MapIntOld(m, z, dimID, pz, rfSize, rfStep)

% This is the way FHLib laid out cells, tiling from the top left corner.

% rfSize = unit width in previous layer units.
% rfStep = unit step size in previous layer units.

[d2, name] = cns_finddim(m, z, dimID, 1);

d1 = cns_finddim(m, pz, name, 1);

if rfSize >= cns_intmax

    m.layers{z}.size{d2}          = 1;
    m.layers{z}.([name '_start']) = 0.5;
    m.layers{z}.([name '_space']) = 1;

    return;

end

if (rfSize < 1) || (mod(rfSize, 1) ~= 0)
    error('rfSize must be a positive integer');
end
if (rfStep < 1) || (mod(rfStep, 1) ~= 0)
    error('rfStep must be a positive integer');
end

pSize  = m.layers{pz}.size{d1};
pStart = m.layers{pz}.([name '_start']);
pSpace = m.layers{pz}.([name '_space']);

nSize = floor((pSize - rfSize) / rfStep) + 1;

if nSize < 1
    nSize  = 0;
    nStart = 0.5;
else
    nStart = pStart + pSpace * (rfSize - 1) / 2;
end

nSpace = pSpace * rfStep;

m.layers{z}.size{d2}          = nSize;
m.layers{z}.([name '_start']) = nStart;
m.layers{z}.([name '_space']) = nSpace;

return;

%***********************************************************************************************************************

function m = MapIntTD(m, z, dimID, pz, rfSize, rfStep)

[d2, name] = cns_finddim(m, z, dimID, 1);

d1 = cns_finddim(m, pz, name, 1);

if (rfSize < 1) || (mod(rfSize, 1) ~= 0)
    error('rfSize must be a positive integer');
end
if (rfStep < 1) || (mod(rfStep, 1) ~= 0)
    error('rfStep must be a positive integer');
end

pSize  = m.layers{pz}.size{d1};
pStart = m.layers{pz}.([name '_start']);
pSpace = m.layers{pz}.([name '_space']);

nSpace = pSpace / rfStep;

if pSize < 1
    nSize  = 0;
    nStart = 0.5;
else
    nSize  = rfSize + (pSize - 1) * rfStep;
    nStart = pStart - 0.5 * (rfSize - 1) * nSpace;
end

m.layers{z}.size{d2}          = nSize;
m.layers{z}.([name '_start']) = nStart;
m.layers{z}.([name '_space']) = nSpace;

return;

%***********************************************************************************************************************

function m = MapWin(m, z, dimID, pz, rfSize, rfStep, rfMargin)

% rfSize   = window width in previous layer units (can be fractional).
% rfStep   = window step size in previous layer units (can be fractional).
% rfMargin = size of margin in previous layer units (can be fractional and/or negative).

[d2, name] = cns_finddim(m, z, dimID, 1);

if rfSize >= cns_intmax

    m.layers{z}.size{d2}          = 1;
    m.layers{z}.([name '_start']) = 0.5;
    m.layers{z}.([name '_space']) = 1;

    return;

end

if rfSize <= 0
    error('rfSize must be positive');
end
if rfStep <= 0
    error('rfStep must be positive');
end

pSpace = m.layers{pz}.([name '_space']);

rfSize   = rfSize   * pSpace;
rfStep   = rfStep   * pSpace;
rfMargin = rfMargin * pSpace;

nSize = 1 + 2 * floor((1 - rfSize - 2 * rfMargin) / (2 * rfStep));

if nSize < 1
    nSize  = 0;
    nStart = 0.5;
else
    nStart = 0.5 - rfStep * (nSize - 1) / 2;
end

m.layers{z}.size{d2}          = nSize;
m.layers{z}.([name '_start']) = nStart;
m.layers{z}.([name '_space']) = rfStep;

return;

%***********************************************************************************************************************

function m = MapTemp1(m, z, dimID, numFrames)

[d2, name] = cns_finddim(m, z, dimID, 2);

if (numFrames < 1) || (mod(numFrames, 1) ~= 0)
    error('numFrames must be a positive integer');
end

m.layers{z}.size{d2}             = numFrames;
m.layers{z}.([name '_start'   ]) = 0.5 - numFrames;
m.layers{z}.([name '_space'   ]) = 1;
m.layers{z}.([name '_lag'     ]) = 0.5;
m.layers{z}.([name '_maxShift']) = numFrames;
m.layers{z}.([name '_total'   ]) = 0;
m.layers{z}.([name '_shift'   ]) = 0;

m.([name '_current']) = 0;

return;

%***********************************************************************************************************************

function m = MapTemp2(m, z, dimID, pz, rfSize, rfStep, nSize)

% rfSize = unit width in previous layer units.
% rfStep = unit step size in previous layer units.
% nSize  = number of output units.

[d2, name] = cns_finddim(m, z, dimID, 2);

d1 = cns_finddim(m, pz, name, 2);

if (rfSize < 1) || (mod(rfSize, 1) ~= 0)
    error('rfSize must be a positive integer');
end
if (rfStep < 1) || (mod(rfStep, 1) ~= 0)
    error('rfStep must be a positive integer');
end
if (nSize < 1) || (mod(nSize, 1) ~= 0)
    error('nSize must be a positive integer');
end

pSize  = m.layers{pz}.size{d1};
pStart = m.layers{pz}.([name '_start']);
pSpace = m.layers{pz}.([name '_space']);
pLag   = m.layers{pz}.([name '_lag'  ]);

if pSize < rfSize
    error('previous layer size is smaller than rfSize');
end

nSpace = pSpace * rfStep;

pNext = pStart + pSpace * pSize;
nNext = pNext + pSpace * (rfSize - 1) / 2;
nStart = nNext - nSpace * nSize;

nLag = pLag + pSpace * (rfSize - 1) / 2;
nMax = min(floor((pSize - rfSize) / rfStep) + 1, nSize);

m.layers{z}.size{d2}             = nSize;
m.layers{z}.([name '_start'   ]) = nStart;
m.layers{z}.([name '_space'   ]) = nSpace;
m.layers{z}.([name '_lag'     ]) = nLag;
m.layers{z}.([name '_maxShift']) = nMax;
m.layers{z}.([name '_total'   ]) = 0;
m.layers{z}.([name '_shift'   ]) = 0;

return;