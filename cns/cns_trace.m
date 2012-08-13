function varargout = cns_trace(m, z, path, varargin)

% CNS_TRACE
%    Click <a href="matlab: cns_help('cns_trace')">here</a> for help.

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

is = cns_iconv(m, z, varargin{:});
is = unique(is(:));

nc = z;

for i = 1 : numel(path)

    if isempty(is), break; end

    nn = path(i);

    if nn < 0

        nn = -nn;

        if ~isfield(m.layers{nc}, 'synapseIs')
            is = [];
        else
            synIs = m.layers{nc}.synapseIs(:, is);
            synZs = m.layers{nc}.synapseZs(:, is);
            is = unique(synIs(synZs(:) == nn));
        end

    else

        if ~isfield(m.layers{nn}, 'synapseIs')
            is = [];
        else
            syns = (m.layers{nn}.synapseZs == nc) & ismember(m.layers{nn}.synapseIs, is);
            is = find(any(syns(:, :), 1)');
        end

    end

    nc = nn;

end

if isempty(is), is = []; end

[varargout{1 : nargout}] = cns_iconv(m, nc, is);

return;
