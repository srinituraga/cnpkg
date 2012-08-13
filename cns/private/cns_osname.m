function [os, bits] = cns_osname

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

comp = upper(computer);

if ~isempty(strmatch('GLN', comp))
    os = 'linux';
elseif ~isempty(strmatch('PC', comp))
    os = 'win';
elseif ~isempty(strmatch('MAC', comp))
    os = 'mac';
elseif ~isempty(strmatch('SOL', comp))
    os = 'sol';
else
    error('unknown architecture');
end

if (numel(comp) >= 2) && strcmp(comp(end - 1 : end), '64')
    bits = 64;
else
    bits = 32;
end

return;