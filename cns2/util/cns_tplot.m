function cns_tplot(data, dt, varargin)

% CNS_TPLOT
%    Click <a href="matlab: cns_help('cns_tplot')">here</a> for help.

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

extra = varargin;

if (numel(extra) >= 1) && isnumeric(extra{1})
    indices = extra{1};
    extra = extra(2 : end);
else
    indices = [];
end

switch numel(indices)
case 0
    first = 1;
    step  = 1;
    last  = Inf;
case 1
    first = 1;
    step  = round(indices(1) / dt);
    last  = Inf;
case 2
    first = round(indices(1) / dt);
    step  = 1;
    last  = round(indices(2) / dt);
case 3
    first = round(indices(1) / dt);
    step  = round(indices(2) / dt);
    last  = round(indices(3) / dt);
otherwise
    error('too many indices');
end

first = max(first, 1);
step  = max(step , 1);
last  = min(last , size(data, 1));

xs = (first : step : last) * dt;
ys = data(first : step : last, :);

plot(xs, ys, extra{:});

return;