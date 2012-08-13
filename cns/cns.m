function varargout = cns(mode, varargin)

% CNS
%    Click <a href="matlab: cns_help('cns')">here</a> for help.

%***********************************************************************************************************************

% Copyright (C) 2010 by Jim Mutch (www.jimmutch.com).
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

if ~mislocked
    mlock;
end

persistent State;
if isempty(State)
    State = struct;
    State.args    = {};
    State.gs      = {};
    State.multi   = ~isempty(which('cns_sync'));
    State.version = cns_version;
end

if ~isempty(varargin) && isa(varargin{1}, 'uint32')
    sid = double(varargin{1}(:)');
    varargin = varargin(2 : end);
else
    sid = -1;
end

State.err = '';

try
    switch mode
    case 'platform', [State, varargout{1 : nargout}] = Platform(State, sid, varargin{:});
    case 'init'    , [State, varargout{1 : nargout}] = Init    (State, sid, varargin{:});
    case 'test'    , [       varargout{1 : nargout}] = Test    (State, sid, varargin{:});
    case 'done'    , [State, varargout{1 : nargout}] = Done    (State, sid, varargin{:});
    case 'run'     , [State, varargout{1 : nargout}] = Run     (State, sid, varargin{:});
    case 'step'    , [State, varargout{1 : nargout}] = Step    (State, sid, varargin{:});
    case 'output'  , [       varargout{1 : nargout}] = Output  (State, sid, varargin{:});
    case 'get'     , [       varargout{1 : nargout}] = Get     (State, sid, varargin{:});
    case 'update'  , [       varargout{1 : nargout}] = Update  (State, sid, varargin{:});
    case 'set'     , [State, varargout{1 : nargout}] = Set     (State, sid, varargin{:});
    case 'shift'   , [State, varargout{1 : nargout}] = Shift   (State, sid, varargin{:});
    case 'wait'    , [       varargout{1 : nargout}] = Wait    (State, sid, varargin{:});
    case 'poll'    , [       varargout{1 : nargout}] = Poll    (State, sid, varargin{:});
    case 'sessions', [       varargout{1 : nargout}] = Sessions(State, sid, varargin{:});
    case 'devcount', [       varargout{1 : nargout}] = DevCount(State, sid, varargin{:});
    otherwise      , error('invalid mode');
    end
catch
    State.err = cns_error;
end

if ~isempty(State.err), error(State.err); end

end

%***********************************************************************************************************************

function [State, platform, rest] = Platform(State, sid, varargin)

if ~isequal(sid, -1), error('session id not allowed'); end
if (nargout < 2) && (nargin < 3), error('not enough arguments'); end

if nargin < 3, varargin = State.args; end

p = GetPlatform(varargin{:});

if nargin >= 3, State.args = varargin; end

if nargout >= 2
    platform = p.args{1};
    rest     = p.args{2};
end

end

%***********************************************************************************************************************

function [State, out] = Init(State, sid, m, varargin)

if (nargout >= 2) && isequal(sid, -1), sid = 0; end

G = GetSession(State, sid);

[G, err] = Init2(State, G, m, varargin{:});

State = SaveSession(State, G);

if nargout >= 2, out = uint32(G.sid); end

State.err = err;

end

%***********************************************************************************************************************

function out = Test(State, sid, m, varargin)

if ~isequal(sid, -1), error('session id not allowed'); end

G = GetSession(State, 0);

[G, err, dump] = Init2(State, G, m, varargin{:});

if isempty(err)
    G = Done2(G);
end

if ~isempty(dump)
    if nargout < 1
        cns_dump(dump, 'summary');
    else
        out = dump;
    end
end

if ~isempty(err)
    if (nargout >= 1) && ~isempty(dump)
        warning(err);
    else
        error(err);
    end
end

end

%***********************************************************************************************************************

function State = Done(State, sids, flag)

if nargin >= 3
    if ~isequal(sids, -1), error('too many arguments'); end
    if ~strcmp(flag, 'all'), error('invalid parameter'); end
    sids = AllSessions(State);
end

for sid = sids

    G = GetSession(State, sid);

    G = Done2(G);

    State = SaveSession(State, G);

end

end

%***********************************************************************************************************************

function [G, err, dump] = Init2(State, G, m, varargin)

dump = [];

try

    if nargin < 3, error('not enough arguments'); end
    if nargin < 4, varargin = State.args; end

    p = GetPlatform(varargin{:});

    g.sid = G.sid;

    if isempty(G.obj)

        if strcmp(p.platform, 'cuda') && ~State.multi && (g.sid > 1)
            if ispc
                error('CNS does not support multiple GPU sessions under Windows');
            else
                error('libraries needed to support multiple GPU sessions were not found on your system');
            end
        end

        % Here is where we used to clear the mex function from the previous session (whose name we remembered) if it
        % was different from the new one.  Now that multiple sessions are supported, we can't do that.

        % Clearing seemed to be necessary to keep CUDA from getting confused.  But we couldn't do it at 'done' time
        % because that would cause CUDA to lose track of its open file handles to the various devices, resulting in
        % 'too many open files' errors after repeated 'init' and 'done' cycles.  However, now that texture references
        % and constants contain the package name, and thus are guaranteed different between MEX files, clearing no
        % longer seems to be necessary.

        g.platform = p.platform;
        g.deviceNo = p.deviceNo;
        g.nice     = p.nice;
        g.debug    = p.debug;
        g.funcName = [m.package '_cns_compiled_' p.platform];
        g.func     = str2func(g.funcName);

        version = g.func();
        if version ~= State.version
            error('function "%s" version (%u) does not match CNS version (%u); use cns_build to recompile package', ...
                g.funcName, version, State.version);
        end

        [g.obj, g.pr] = g.func(0, p.deviceNo, double(p.nice), double(p.debug));

    else

        if ~strcmp(m.package, G.m.package), error('to reinitialize for a new package, call "done" first'); end
        if ~strcmp(p.platform, G.platform), error('to reinitialize for a new platform, call "done" first'); end
        if p.deviceNo ~= G.deviceNo, error('to reinitialize for a new device, call "done" first'); end
        if p.nice ~= G.nice, error('to reinitialize for a new nice value, call "done" first'); end
        if p.debug ~= G.debug, error('to reinitialize for a new platform, call "done" first'); end

        G.func(3, G.obj);

        g.platform = G.platform;
        g.deviceNo = G.deviceNo;
        g.nice     = G.nice;
        g.debug    = G.debug;
        g.funcName = G.funcName;
        g.func     = G.func;
        g.obj      = G.obj;
        g.pr       = G.pr;

    end

    G = g;

    G.def = cns_def(m);

    if ismember('init', G.def.methods)
        m = feval([m.package '_cns'], 'init', m);
    end

    G.m = MemModel(m, G.def);

    [G.s, h] = InitStart;
    [G.s, h] = InitGlobal     (G, G.s, h, m, G.def);
    [G.s, h] = InitGroups     (G, G.s, h, m, G.def);
    [G.s, h] = InitLayers     (G, G.s, h, m, G.def);
    [G.s, h] = ReadAllConsts  (G, G.s, h, m, G.def);
    [G.s, h] = ReadAllArrays  (G, G.s, h, m, G.def);
    [G.s, h] = ReadAllTextures(G, G.s, h, m, G.def);
    [G.s, h] = ReadAllCommon  (G, G.s, h, m, G.def);
    [G.s, h] = ReadAllNFields (G, G.s, h, m, G.def);
    [G.s, h] = ReadAllSynapses(G, G.s, h, m, G.def);
    [G.s, h] = ReadAllSFields (G, G.s, h, m, G.def);
    [G.s, h] = FinalizeGlobal (G, G.s, h, m, G.def);
    [G.s, h] = FinalizeLayers (G, G.s, h, m, G.def);
    [G.s, h] = MakeKernels    (G, G.s, h, m, G.def);
    [G.s, h] = Finalize       (G, G.s, h);

    G.user = struct;

    if nargout >= 3
        dump = struct;
        dump.G = G;
        dump.h = h;
    end

    G.func(2, G.obj, CB(G), G.s, h);

    err = '';

catch

    err = cns_error;

    G = Done2(G);

end

if ~isempty(dump), dump.err = err; end

end

%***********************************************************************************************************************

function G = Done2(G)

if isempty(G.obj), return; end

G.func(1, G.obj);

G.obj = [];

end

%***********************************************************************************************************************

function p = GetPlatform(platstr, niceFlag)

if nargin < 1, platstr = 'cuda'; end

pos = find(isstrprop(platstr, 'digit'), 1);
if isempty(pos)
    deviceNo = -1;
else
    if ~all(isstrprop(platstr(pos + 1 : end), 'digit')), error('invalid device number'); end
    deviceNo = str2double(platstr(pos : end));
    if isnan(deviceNo), error('invalid device number'); end
    platstr = platstr(1 : pos - 1);
end

switch lower(platstr)
case {'cuda', 'gpu'}
    p.platform = 'cuda';
    p.debug    = false;
case 'cpu'
    p.platform = 'cpu';
    p.debug    = false;
case 'debug'
    p.platform = 'cpu';
    p.debug    = true;
otherwise
    error('platform "%s" is invalid', platstr);
end

if strcmp(p.platform, 'cpu') && (deviceNo >= 0), error('device number not valid for cpu'); end
p.deviceNo = deviceNo;

if strcmp(p.platform, 'cuda')
    if nargin < 2, niceFlag = 'nice'; end
    switch lower(niceFlag)
    case 'nice', p.nice = true;
    case 'mean', p.nice = false;
    otherwise  , error('nice flag "%s" is invalid', niceFlag);
    end
else
    if nargin >= 2, error('too many arguments'); end
    p.nice = true;
end

if strcmp(p.platform, 'cuda')
    if p.deviceNo == -1, p.args{1} = 'gpu'; else p.args{1} = sprintf('gpu%u', p.deviceNo); end
    if p.nice, p.args{2} = 'nice'; else p.args{2} = 'mean'; end
else
    if p.debug, p.args{1} = 'debug'; else p.args{1} = 'cpu'; end
    p.args{2} = '';
end

end

%***********************************************************************************************************************

function mm = MemModel(m, def)

mm = struct;

names = cns_reservednames(true, true);

for f = 1 : numel(names)
    if isfield(m, names{f}), mm.(names{f}) = m.(names{f}); end
end

names = cns_reservednames(false, true);

for z = 1 : numel(def.layers)
    for f = 1 : numel(names)
        if isfield(m.layers{z}, names{f}), mm.layers{z}.(names{f}) = m.layers{z}.(names{f}); end
    end
end

for g = 1 : def.gCount
    for f = 1 : numel(names)
        if isfield(m.groups{g}, names{f}), mm.groups{g}.(names{f}) = m.groups{g}.(names{f}); end
    end
end

end

%***********************************************************************************************************************

function [s, h] = InitStart

s = struct;
h = struct;

h.cLayerTable = zeros(0, 1); h.sLayerTable = 0;
h.cMVTable    = zeros(0, 1); h.sMVTable    = 1;
h.cVarMeta    = zeros(0, 1); h.sVarMeta    = 2;

h.dData = zeros(0, 1, 'single');

end

%***********************************************************************************************************************

function [s, h] = InitGlobal(G, s, h, m, def)

s.mvOff = TagOff(numel(h.cMVTable), h.sMVTable);
h.cMVTable(end + 1 : end + 2 * numel(def.list.mvm.syms), 1) = 0;

if isfield(m, 'independent')
    s.independent = double(m.independent);
else
    s.independent = 0;
end

end

%***********************************************************************************************************************

function [s, h] = InitGroups(G, s, h, m, def)

for g = 1 : numel(def.groups)

    d = def.layers{def.groups{g}.zs(1)};
    c = struct;

    c.mvOff = TagOff(numel(h.cMVTable), h.sMVTable);
    h.cMVTable(end + 1 : end + 2 * numel(d.list.mvg.syms), 1) = 0;

    s.groups{g} = c;

end

end

%***********************************************************************************************************************

function [s, h] = InitLayers(G, s, h, m, def)

s.ySizes  = zeros(1, numel(def.layers));
s.xCounts = zeros(1, numel(def.layers));

s.isType = false(numel(def.layers), 0);

for z = 1 : numel(def.layers)

    d = def.layers{z};

    if d.kernel
        if mod(max(d.blockYSize, 0.5), G.pr.blockYSizeAlign) ~= 0
            error('type "%s": block y size must be a multiple of %u', d.type, G.pr.blockYSizeAlign);
        end
        if mod(max(d.blockYSize * d.blockXSize, 0.5), G.pr.blockSizeAlign) ~= 0
            error('type "%s": block size must be a multiple of %u', d.type, G.pr.blockSizeAlign);
        end
        if d.blockYSize * d.blockXSize < G.pr.minBlockSize
            error('type "%s": block size must be at least %u', d.type, G.pr.minBlockSize);
        end
    end

    c = struct;

    c.typeNo = d.typeNo;

    try
        c.t = cns_trans('create', d, m.layers{z}.size, G.pr.blockYSizeAlign);
    catch
        error('z=%u: %s', z, cns_error);
    end
    c.yCount0 = c.t.siz3 (1);
    c.ySize0  = c.t.siz3a(1);
    c.yCount  = c.t.siz4a(1);
    c.ySize   = c.t.siz4b(1);
    c.xCount  = c.t.siz4a(2);

    if c.yCount0 == c.ySize0
        yCountOpt = c.yCount;
    else
        yCountOpt = c.yCount0;
    end

    if ~isfield(m, 'quiet') || ~m.quiet
        if (c.yCount > 0) && (yCountOpt < G.pr.blockYSizeAlign) && (c.xCount >= 2 * G.pr.blockYSizeAlign)
            warning('z=%u: for thin layers, yCount > xCount is more efficient', z);
        end
    end

    if d.kernel
        c.blockYSize = OptimizeBlock(d.blockYSize, d.blockXSize, G.pr.blockYSizeAlign, yCountOpt, c.xCount);
    else
        c.blockYSize = 0;
    end

    c.sFlag = any(isfield(m.layers{z}, {'synapseIs', 'synapseZs', 'synapseTs', ...
        d.cat.sc.syms{:}, d.cat.sv.syms{:}}));

    if c.sFlag
        if d.synTypeNo == 0
            error('z=%u: synapses are not defined for this type', z);
        end
        if ~isfield(m.layers{z}, 'synapseIs')
            error('z=%u: field "synapseIs" is missing', z);
        end
        c.sSize = size(m.layers{z}.synapseIs, 1);
        if c.sSize > 65535
            error('z=%u: at most %u synapses are allowed', z, 65535);
        end
    else
        c.sSize = 0;
    end

    c.st = cns_trans('add', c.t, c.sSize);

    c.mvOff = TagOff(numel(h.cMVTable), h.sMVTable);
    h.cMVTable(end + 1 : end + 2 * numel(d.list.mvl.syms), 1) = 0;

    s.layers{z} = c;

    s.ySizes (z) = c.ySize;
    s.xCounts(z) = c.xCount;
    s.ts     (z) = c.t;

    s.isType(z, 1 : numel(d.isType)) = d.isType;

end

end

%***********************************************************************************************************************

function [s, h] = UpdateGlobal(func, G, s, h, m, def, varargin)

try
    [s, h] = func(G, s, h, m, def, 0, varargin{:});
catch
    error('z=0: %s', cns_error);
end

end

%***********************************************************************************************************************

function [s, h] = UpdateGroups(func, G, s, h, m, def, varargin)

for g = 1 : numel(def.groups)

    z = def.groups{g}.zs(1);

    if g <= def.gCount
        data = m.groups{g};
    else
        data = m.layers{z};
    end

    try
        [s.groups{g}, h] = func(G, s.groups{g}, h, data, def.layers{z}, -g, varargin{:});
    catch
        if g <= def.gCount
            error('group=%u: %s', g, cns_error);
        else
            error('z=%u: %s', z, cns_error);
        end
    end

end

end

%***********************************************************************************************************************

function [s, h] = UpdateLayers(func, G, s, h, m, def, varargin)

for z = 1 : numel(def.layers)

    try
        [s.layers{z}, h] = func(G, s.layers{z}, h, m.layers{z}, def.layers{z}, z, varargin{:});
    catch
        error('z=%u: %s', z, cns_error);
    end

end

end

%***********************************************************************************************************************

function [s, h] = ReadAllConsts(G, s, h, m, def)

h.cData = zeros(0, 1, 'single');

[s, h] = UpdateGlobal(@ReadConsts, G, s, h, m, def, 'mp');
[s, h] = UpdateGroups(@ReadConsts, G, s, h, m, def, 'gp');
[s, h] = UpdateLayers(@ReadConsts, G, s, h, m, def, 'lp');

if numel(h.cData) > G.pr.maxCData
    error('maximum number of constants (%u) exceeded', G.pr.maxCData);
end

end

%***********************************************************************************************************************

function [c, h] = ReadConsts(G, c, h, m, d, n, cat)

c.cOff = numel(h.cData);

data = zeros(numel(d.list.(cat).svSyms), 1, 'single');

for f = 1 : numel(d.list.(cat).svSyms)

    name = d.list.(cat).svSyms{f};
    dd = d.sym.(name);
    pos = dd.pos;

    a = GetField(m, name, [], dd);
    if isfield(dd, 'ptrTypeNo'), CheckPointer(G.s.isType, name, a, dd.ptrTypeNo, true); end

    data(pos, 1) = a;

end

mvOff = RelOff(c.mvOff);

for f = 1 : numel(d.list.(cat).mvSyms)

    name = d.list.(cat).mvSyms{f};
    dd = d.sym.(name);
    pos = 2 * dd.pos - 1;

    a = GetField(m, name, [], dd);
    if isfield(dd, 'ptrTypeNo'), CheckPointer(G.s.isType, name, a, dd.ptrTypeNo, false); end

    h.cMVTable(mvOff + pos    , 1) = numel(h.cData) + numel(data);
    h.cMVTable(mvOff + pos + 1, 1) = numel(a);

    data(end + 1 : end + numel(a), 1) = a;

end

h.cData(end + 1 : end + numel(data), 1) = data;

end

%***********************************************************************************************************************

function [s, h] = ReadAllArrays(G, s, h, m, def)

[s, h] = UpdateGlobal(@ReadArrays, G, s, h, m, def, 'ma');
[s, h] = UpdateGroups(@ReadArrays, G, s, h, m, def, 'ga');
[s, h] = UpdateLayers(@ReadArrays, G, s, h, m, def, 'la');

end

%***********************************************************************************************************************

function [c, h] = ReadArrays(G, c, h, m, d, n, cat)

meta = zeros(0, 1);
data = zeros(0, 1, 'single');

mvOff = RelOff(c.mvOff);

for f = 1 : numel(d.cat.(cat).syms)

    name = d.cat.(cat).syms{f};
    pos = 2 * d.sym.(name).pos - 1;

    [as, siz2ps] = GetArray(m, name, d.sym.(name), G.pr.blockYSizeAlign);

    h.cMVTable(mvOff + pos    , 1) = TagOff(numel(h.cVarMeta) + numel(meta) + 2, h.sVarMeta);
    h.cMVTable(mvOff + pos + 1, 1) = numel(as);

    for i = 1 : numel(as)

        off = numel(h.dData) + numel(data);

        entry = zeros(ceil((2 + numel(siz2ps{i})) / 2) * 2, 1);
        entry(1) = mod(off, 65536);
        entry(2) = floor(off / 65536);
        entry(2 + 1 : 2 + numel(siz2ps{i})) = siz2ps{i}(:);

        meta(end + 1 : end + numel(entry), 1) = entry;
        data(end + 1 : end + numel(as{i}), 1) = as{i}(:);

    end

end

h.cVarMeta(end + 1 : end + numel(meta), 1) = meta;
h.dData   (end + 1 : end + numel(data), 1) = data;

end

%***********************************************************************************************************************

function [s, h] = ReadAllTextures(G, s, h, m, def)

h.tTexNo   = [];
h.tLayerNo = [];
h.tValueNo = [];
h.tSiz2p   = {};
h.tArray   = {};

[s, h] = UpdateGlobal(@ReadTextures1, G, s, h, m, def, 'mt');
[s, h] = UpdateGroups(@ReadTextures1, G, s, h, m, def, 'gt');
[s, h] = UpdateLayers(@ReadTextures1, G, s, h, m, def, 'lt');

h.tYOff = zeros(1, numel(h.tTexNo));
h.tXOff = zeros(1, numel(h.tTexNo));

h.tTData = cell(def.tCount, 1);

for t = 1 : def.tCount

    inds = find(h.tTexNo == t);

    [yCounts, xCounts] = cellfun(@size, h.tArray(inds));
    [tyCount, txCount, yOffs, xOffs] = cns_textile(G.pr, yCounts, xCounts);
    h.tYOff(inds) = yOffs;
    h.tXOff(inds) = xOffs;

    h.tTData{t} = zeros(tyCount, txCount, 'single');

    for i = 1 : numel(inds)
        h.tTData{t}(yOffs(i) + 1 : yOffs(i) + yCounts(i), xOffs(i) + 1 : xOffs(i) + xCounts(i)) = h.tArray{inds(i)};
    end

end

h = rmfield(h, 'tArray');

[s, h] = UpdateGlobal(@ReadTextures2, G, s, h, m, def, 'mt');
[s, h] = UpdateGroups(@ReadTextures2, G, s, h, m, def, 'gt');
[s, h] = UpdateLayers(@ReadTextures2, G, s, h, m, def, 'lt');

h = rmfield(h, {'tTexNo', 'tLayerNo', 'tValueNo', 'tSiz2p', 'tYOff', 'tXOff'});

end

%***********************************************************************************************************************

function [c, h] = ReadTextures1(G, c, h, m, d, n, cat)

for f = 1 : numel(d.cat.(cat).syms)

    name = d.cat.(cat).syms{f};

    [as, siz2ps] = GetArray(m, name, d.sym.(name), []);

    for i = 1 : numel(as)
        h.tTexNo  (end + 1) = d.sym.(name).resNo;
        h.tLayerNo(end + 1) = n;
        h.tValueNo(end + 1) = i;
        h.tSiz2p  (end + 1) = siz2ps(i);
        h.tArray  (end + 1) = as(i);
    end

end

end

%***********************************************************************************************************************

function [c, h] = ReadTextures2(G, c, h, m, d, n, cat)

meta = zeros(0, 1);

mvOff = RelOff(c.mvOff);

for f = 1 : numel(d.cat.(cat).syms)

    name = d.cat.(cat).syms{f};
    pos = 2 * d.sym.(name).pos - 1;

    inds = ((h.tTexNo == d.sym.(name).resNo) & (h.tLayerNo == n));
    vCount = sum(inds);

    h.cMVTable(mvOff + pos    , 1) = TagOff(numel(h.cVarMeta) + numel(meta) + 2, h.sVarMeta);
    h.cMVTable(mvOff + pos + 1, 1) = vCount;

    for i = 1 : vCount

        ind = find(inds & (h.tValueNo == i));

        entry = zeros(ceil((2 + numel(h.tSiz2p{ind})) / 2) * 2, 1);
        entry(1) = h.tYOff(ind);
        entry(2) = h.tXOff(ind);
        entry(2 + 1 : 2 + numel(h.tSiz2p{ind})) = h.tSiz2p{ind}(:);

        meta(end + 1 : end + numel(entry), 1) = entry;

    end

end

h.cVarMeta(end + 1 : end + numel(meta), 1) = meta;

end

%***********************************************************************************************************************

function [s, h] = ReadAllCommon(G, s, h, m, def)

h.tResNo   = [];
h.tLayerNo = [];
h.tArray   = {};

[s, h] = UpdateLayers(@ReadCommon1, G, s, h, m, def);

h.tYOff = zeros(1, numel(h.tResNo));
h.tXOff = zeros(1, numel(h.tResNo));

s.ccYSizes  = zeros(1, def.ccCount);
s.ccXCounts = zeros(1, def.ccCount);
s.cvYSizes  = zeros(1, def.cvCount);
s.cvXCounts = zeros(1, def.cvCount);

h.tCData = cell(def.ccCount, 1);
h.tVData = cell(def.cvCount, 1);

for r = 1 : def.ccCount
    inds = find(h.tResNo == -r);
    ySizes  = s.ySizes (h.tLayerNo(inds));
    xCounts = s.xCounts(h.tLayerNo(inds));
    [s.ccYSizes(r), s.ccXCounts(r), yOffs, xOffs] = cns_textile(G.pr, ySizes, xCounts);
    h.tYOff(inds) = yOffs;
    h.tXOff(inds) = xOffs;
    h.tCData{r} = zeros(s.ccYSizes(r), s.ccXCounts(r), 'single');
    for i = 1 : numel(inds)
        h.tCData{r}(yOffs(i) + 1 : yOffs(i) + ySizes(i), xOffs(i) + 1 : xOffs(i) + xCounts(i)) = h.tArray{inds(i)};
    end
end

for r = 1 : def.cvCount
    inds = find(h.tResNo == r);
    ySizes  = s.ySizes (h.tLayerNo(inds));
    xCounts = s.xCounts(h.tLayerNo(inds));
    [s.cvYSizes(r), s.cvXCounts(r), yOffs, xOffs] = cns_textile(G.pr, ySizes, xCounts);
    h.tYOff(inds) = yOffs;
    h.tXOff(inds) = xOffs;
    h.tVData{r} = zeros(s.cvYSizes(r), s.cvXCounts(r), 'single');
    for i = 1 : numel(inds)
        h.tVData{r}(yOffs(i) + 1 : yOffs(i) + ySizes(i), xOffs(i) + 1 : xOffs(i) + xCounts(i)) = h.tArray{inds(i)};
    end
end

h = rmfield(h, 'tArray');

[s, h] = UpdateLayers(@ReadCommon2, G, s, h, m, def);

h = rmfield(h, {'tResNo', 'tLayerNo', 'tYOff', 'tXOff'});

end

%***********************************************************************************************************************

function [c, h] = ReadCommon1(G, c, h, m, d, n)

for f = 1 : numel(d.cat.cc.syms)
    name = d.cat.cc.syms{f};
    h.tResNo  (end + 1) = -d.sym.(name).resNo;
    h.tLayerNo(end + 1) = n;
    h.tArray  {end + 1} = GetField(m, name, c.t, d.sym.(name));
end

for f = 1 : numel(d.cat.cv.syms)
    name = d.cat.cv.syms{f};
    h.tResNo  (end + 1) = d.sym.(name).resNo;
    h.tLayerNo(end + 1) = n;
    h.tArray  {end + 1} = GetField(m, name, c.t, d.sym.(name));
end

end

%***********************************************************************************************************************

function [c, h] = ReadCommon2(G, c, h, m, d, n)

meta = zeros(2, numel(d.list.c.syms));

for f = 1 : numel(d.cat.cc.syms)
    name = d.cat.cc.syms{f};
    pos = d.sym.(name).pos;
    ind = find((h.tResNo == -d.sym.(name).resNo) & (h.tLayerNo == n));
    meta(1, pos) = h.tYOff(ind);
    meta(2, pos) = h.tXOff(ind);
end

for f = 1 : numel(d.cat.cv.syms)
    name = d.cat.cv.syms{f};
    pos = d.sym.(name).pos;
    ind = find((h.tResNo == d.sym.(name).resNo) & (h.tLayerNo == n));
    meta(1, pos) = h.tYOff(ind);
    meta(2, pos) = h.tXOff(ind);
end

c.tOff = TagOff(numel(h.cVarMeta), h.sVarMeta);

h.cVarMeta(end + 1 : end + numel(meta), 1) = meta(:);

end

%***********************************************************************************************************************

function [s, h] = ReadAllNFields(G, s, h, m, def)

h.dWData = zeros(0, 1, 'single');

[s, h] = UpdateLayers(@ReadNFields, G, s, h, m, def);

end

%***********************************************************************************************************************

function [c, h] = ReadNFields(G, c, h, m, d, n)

c.ndOff = numel(h.dData);

data = zeros(c.ySize, c.xCount, numel(d.list.n.svSyms), 'single');

for f = 1 : numel(d.list.n.svSyms)

    name = d.list.n.svSyms{f};
    pos = d.sym.(name).pos;

    a = GetField(m, name, c.t, d.sym.(name));

    data(:, :, pos) = a;

end

mvOff = RelOff(c.mvOff);

for f = 1 : numel(d.list.n.mvSyms)

    name = d.list.n.mvSyms{f};
    pos = 2 * d.sym.(name).pos - 1;

    [a, vc] = GetField(m, name, c.t, d.sym.(name));

    if size(data, 3) > 65535
        error('maximum number of values (%u) exceeded', 65535);
    end

    h.cMVTable(mvOff + pos    , 1) = size(data, 3);
    h.cMVTable(mvOff + pos + 1, 1) = vc;

    data(:, :, end + 1 : end + vc) = a;

end

h.dData(end + 1 : end + numel(data), 1) = data(:);

c.nwOff = numel(h.dWData);

data = zeros(c.ySize, c.xCount, numel(d.cat.nw.svSyms), 'single');

for f = 1 : numel(d.cat.nw.svSyms)

    name = d.cat.nw.svSyms{f};
    pos = d.sym.(name).pos;

    a = GetField(m, name, c.t, d.sym.(name));

    data(:, :, pos) = a;

end

h.dWData(end + 1 : end + numel(data), 1) = data(:);

end

%***********************************************************************************************************************

function [s, h] = ReadAllSynapses(G, s, h, m, def)

h.dNeurons  = zeros(0, 1, 'uint32');
h.dSynapses = zeros(0, 1, 'uint16');

[s, h] = UpdateLayers(@ReadSynapses, G, s, h, m, def);

end

%***********************************************************************************************************************

function [c, h] = ReadSynapses(G, c, h, m, d, n)

c.nmOff = numel(h.dNeurons );
c.smOff = numel(h.dSynapses) / 4;

if c.sFlag

    if ~cns_trans('sizeis', c.st, m.synapseIs)
        error('synapseIs is incorrectly sized');
    end

    if ~isfield(m, 'synapseZs')
        error('field "synapseZs" is missing');
    end
    if numel(m.synapseZs) == 1
        synapseZs = uint16(m.synapseIs ~= 0) * uint16(m.synapseZs);
    else
        if ~cns_trans('sizeis', c.st, m.synapseZs)
            error('synapseZs must be the same size as synapseIs');
        end
        synapseZs = m.synapseZs;
    end

    if isfield(m, 'synapseTs')
        if numel(m.synapseTs) == 1
            synapseTs = uint16(m.synapseIs ~= 0) * uint16(m.synapseTs);
        else
            if ~cns_trans('sizeis', c.st, m.synapseTs)
                error('synapseTs must be the same size as synapseIs');
            end
            synapseTs = m.synapseTs;
        end
    else
        synapseTs = uint16(m.synapseIs ~= 0);
    end

    % TODO: roll in 'pack' call may not be zero?
    [neurons, synapses] = cns_initsynapses(CB(G), n, ...
        cns_trans('pack', c.st, uint32(m.synapseIs), false, 0), ...
        cns_trans('pack', c.st, uint16(  synapseZs), false, 0), ...
        cns_trans('pack', c.st, uint16(  synapseTs), false, 0), ...
        c.ySize, G.s.ts, ...
        uint32(G.s.isType(:, d.synTypeNo)));

elseif d.synTypeNo ~= 0

    neurons  = zeros(1, c.ySize, c.xCount, 'uint32');
    synapses = zeros(0, 'uint16');

else

    neurons  = zeros(0, 'uint32');
    synapses = zeros(0, 'uint16');

end

h.dNeurons (end + 1 : end + numel(neurons ), 1) = neurons (:);
h.dSynapses(end + 1 : end + numel(synapses), 1) = synapses(:);

end

%***********************************************************************************************************************

function [s, h] = ReadAllSFields(G, s, h, m, def)

[s, h] = UpdateLayers(@ReadSFields, G, s, h, m, def);

end

%***********************************************************************************************************************

function [c, h] = ReadSFields(G, c, h, m, d, n)

c.sdOff = numel(h.dData);

if c.sFlag

    data = zeros(c.ySize, c.xCount, c.sSize, numel(d.list.s.svSyms), 'single');

    for f = 1 : numel(d.list.s.svSyms)

        name = d.list.s.svSyms{f};
        pos = d.sym.(name).pos;

        a = GetField(m, name, c.st, d.sym.(name));

        data(:, :, :, pos) = a;

    end

    mvOff = RelOff(c.mvOff);

    for f = 1 : numel(d.list.s.mvSyms)

        name = d.list.s.mvSyms{f};
        pos = 2 * d.sym.(name).pos - 1;

        [a, vc] = GetField(m, name, c.st, d.sym.(name));

        if size(data, 4) > 65535
            error('maximum number of values (%u) exceeded', 65535);
        end

        h.cMVTable(mvOff + pos    , 1) = size(data, 4);
        h.cMVTable(mvOff + pos + 1, 1) = vc;

        data(:, :, :, end + 1 : end + vc) = a;

    end

    h.dData(end + 1 : end + numel(data), 1) = data(:);

end

end

%***********************************************************************************************************************

function [s, h] = FinalizeGlobal(G, s, h, m, def)

h.sn = def.shiftName;

if isempty(h.sn)
    h.sh.current   = 0;
    s.shCurrentOff = 0;
else
    h.sh.current(1) = m.([h.sn '_current']);
    s.shCurrentOff  = s.cOff + def.sym.([h.sn '_current']).pos - 1;
end

end

%***********************************************************************************************************************

function [s, h] = FinalizeLayers(G, s, h, m, def)

for z = 1 : numel(def.layers)
    g = def.layers{z}.g;
    s.layers{z}.gmvOff = s.groups{g}.mvOff;
    s.layers{z}.gcOff  = s.groups{g}.cOff;
end

h.sh.flag     = zeros(1, numel(def.layers));
h.sh.size     = ones (1, numel(def.layers));
h.sh.start    = zeros(1, numel(def.layers));
h.sh.space    = ones (1, numel(def.layers));
h.sh.lag      = zeros(1, numel(def.layers));
h.sh.maxShift = zeros(1, numel(def.layers));
h.sh.total    = zeros(1, numel(def.layers));
h.sh.shift    = zeros(1, numel(def.layers));

[s, h] = UpdateLayers(@FinalizeLayersSub, G, s, h, m, def);

h.sh.roll = ComputeRoll(h.sh.total, h.sh.size);

end

%***********************************************************************************************************************

function [c, h] = FinalizeLayersSub(G, c, h, m, d, z)

% Make layer table entry.

e = cns_consts('layertable', G.def.maxSiz2p);

entry = zeros(1, e.len);

entry(e.gmvOff + 1) = c.gmvOff;
entry(e.mvOff  + 1) = c.mvOff;
entry(e.gcOff  + 1) = c.gcOff;
entry(e.cOff   + 1) = c.cOff;
entry(e.tOff   + 1) = c.tOff;
entry(e.xCount + 1) = c.xCount;
entry(e.nwOff  + 1) = mod(c.nwOff, 65536);
entry(e.nwOff  + 2) = floor(c.nwOff / 65536);
entry(e.ndOff  + 1) = mod(c.ndOff, 65536);
entry(e.ndOff  + 2) = floor(c.ndOff / 65536);

siz2p = cns_trans('siz2p', c.t);
if any(siz2p > 65535), error('size dimension exceeds %u', 65535); end
entry(e.siz2p + 1 : e.siz2p + numel(siz2p)) = siz2p;

% Any further entry is the initial roll amount for any shiftable dimension (implicitly 0).
rollOff = numel(h.cLayerTable) + e.siz2p + numel(siz2p);

h.cLayerTable(end + 1 : end + numel(entry), 1) = entry';

% Get shift information.

i = find(d.dmap == 2, 1);

if isempty(i)

    c.xShift = 0;

    c.shStartOff = 0;
    c.shShiftOff = 0;
    c.shRollOff  = 0;

else

    c.xShift = c.xCount / m.size{i}; % columns per shiftable coordinate

    h.sh.flag    (z) = 1;
    h.sh.size    (z) = m.size{i};
    h.sh.start   (z) = m.([h.sn '_start'   ]);
    h.sh.space   (z) = m.([h.sn '_space'   ]);
    h.sh.lag     (z) = m.([h.sn '_lag'     ]);
    h.sh.maxShift(z) = m.([h.sn '_maxShift']);
    h.sh.total   (z) = m.([h.sn '_total'   ]);
    h.sh.shift   (z) = m.([h.sn '_shift'   ]);

    c.shStartOff = c.cOff + d.sym.([h.sn '_start']).pos - 1;
    c.shShiftOff = c.cOff + d.sym.([h.sn '_shift']).pos - 1;
    c.shRollOff  = rollOff;

end

end

%***********************************************************************************************************************

function [s, h] = MakeKernels(G, s, h, m, def)

keys  = zeros(0, 7);
keyzs = zeros(1, 0);

for z = 1 : numel(def.layers)

    if z == 1
        usingSteps = isfield(m.layers{z}, 'stepNo');
    elseif isfield(m.layers{z}, 'stepNo') ~= usingSteps
        error('z=%u: either all layers must specify stepNo, or no layers', z);
    end
    if isfield(m.layers{z}, 'stepNo')
        r = m.layers{z}.stepNo(:)';
        if def.layers{z}.kernel
            if any(r < 0) || any(mod(r, 1) ~= 0), error('z=%u: invalid stepNo', z); end
            if numel(unique(r(r ~= 0))) ~= sum(r ~= 0), error('z=%u: stepNo values must be unique', z); end
        else
            if any(r ~= 0), error('z=%u: invalid stepNo', z); end
        end
    else
        if def.layers{z}.kernel
            r = 1;
        else
            r = [];
        end
    end
    p = find(r ~= 0);
    r = r(p);

    for i = 1 : numel(r)
        keys(end + 1, :) = [r(i), def.layers{z}.typeNo, p(i), ...
            h.sh.size(z), h.sh.start(z), h.sh.space(z), h.sh.lag(z)];
        keyzs(end + 1) = z;
    end

end

[keys, ans, inds] = unique(keys, 'rows');
rs = keys(:, 1)';
ps = keys(:, 3)';
ns = max(keys(:, 4)', 1);
zs = cell(1, numel(rs));
for k = 1 : numel(rs)
    zs{k} = keyzs(inds == k);
end

if ~isequal(unique(rs), 1 : max(rs)), error('stepNo values must be contiguous'); end

h.dBlocks   = zeros(0, 1, 'uint16');
h.hKernelZs = zeros(0, 1, 'uint32');

for k = 1 : numel(rs)

    d = def.layers{zs{k}(1)};
    c = struct;

    c.type      = d.type;
    c.typeNo    = d.typeNo;
    c.stepNo    = rs(k);
    c.phase     = ps(k);
    c.blockSize = d.blockYSize * d.blockXSize;

    blocks = zeros(4, 0, ns(k), 'uint16');

    for z = zs{k}

        blockYSize = s.layers{z}.blockYSize;
        blockXSize = c.blockSize / blockYSize;

        yCount0 = s.layers{z}.yCount0;
        ySize0  = s.layers{z}.ySize0;
        yCount  = s.layers{z}.yCount;
        xCount  = s.layers{z}.xCount;

        ys = 0 : blockYSize : yCount - 1;
        if yCount0 == ySize0
            yc = min(blockYSize, yCount - ys);
        else
            yc = min(blockYSize, yCount0 - mod(ys, ySize0));
        end

        % Make sure all cells in a block have the same coordinate along the shiftable dimension.
        xc = repmat(min(blockXSize, xCount / ns(k) : -blockXSize : 1), 1, ns(k));
        if isempty(xc)
            xs = xc;
        else
            xs = cumsum([0 xc(1 : end - 1)]);
        end

        [xsGrid, ysGrid] = meshgrid(xs, ys);
        xc(xc == 256) = 0;
        yc(yc == 256) = 0;
        xcGrid = repmat(xc , numel(ys), 1);
        ycGrid = repmat(yc', 1, numel(xs));
        ccGrid = xcGrid * 256 + ycGrid;

        blocks2 = zeros(4, numel(xsGrid) / ns(k), ns(k), 'uint16');
        blocks2(1, :, :) = reshape(xsGrid, 1, [], ns(k));
        blocks2(2, :, :) = reshape(ysGrid, 1, [], ns(k));
        blocks2(3, :, :) = z - 1;
        blocks2(4, :, :) = reshape(ccGrid, 1, [], ns(k));

        % Eliminate partition camping.  Currently disabled because it seems to slow things down slightly.
        % inds = Uncamp(numel(ys), numel(xs) / ns(k));
        % blocks2 = blocks2(:, inds, :);

        blocks = cat(2, blocks, blocks2);

    end

    c.bOff   = numel(h.dBlocks) / 4;
    c.bCount = numel(blocks) / 4;

    if any(d.dmap == 2)
        c.bShift = size(blocks, 2); % blocks per shiftable coordinate
    else
        c.bShift = 0;
    end

    h.dBlocks(end + 1 : end + numel(blocks), 1) = blocks(:);

    c.zOff   = numel(h.hKernelZs);
    c.zCount = numel(zs{k});

    h.hKernelZs(end + 1 : end + numel(zs{k}), 1) = zs{k}(:) - 1;

    s.kernels{k} = c;

end

end

%***********************************************************************************************************************

function dids = Uncamp(ny, nx)

% Diagonal block reordering algorithm from "Optimizing Matrix Transpose in CUDA".
% Eliminates the partition camping problem.

lids = 0 : ny * nx - 1;

dxs = mod(lids, nx);
dys = mod(floor(lids / nx) + dxs, ny);

dids = zeros(1, ny * nx);
dids(dxs * ny + dys + 1) = lids + 1;

end

%***********************************************************************************************************************

function [s, h] = Finalize(G, s, h)

s.groups  = [s.groups{:} ];
s.layers  = [s.layers{:} ];
s.kernels = [s.kernels{:}];

offs = cumsum([numel(h.cLayerTable), numel(h.cMVTable)]);

s = Tag2Abs(s, offs);

% Note: this copy of cMeta (s.cMeta) will never be updated after initialization.
s.cMeta = Tag2Abs([h.cLayerTable; h.cMVTable; h.cVarMeta], offs);
if numel(s.cMeta) > G.pr.maxCMeta
    error('maximum number of metadata elements (%u) exceeded', G.pr.maxCMeta);
end
if any(s.cMeta > 65535)
    error('metadata value exceeds %u', 65535);
end
h.cMeta = uint16(s.cMeta);
h = rmfield(h, {'cLayerTable', 'sLayerTable'});
h = rmfield(h, {'cMVTable'   , 'sMVTable'   });
h = rmfield(h, {'cVarMeta'   , 'sVarMeta'   });

% We temporarily stored these in 'h' to avoid them being subject to Tag2Abs above.
s.sn = h.sn;
s.sh = h.sh;
h = rmfield(h, {'sn', 'sh'});

end

%***********************************************************************************************************************

function blockYSize = OptimizeBlock(blockYSize, blockXSize, blockYSizeAlign, yCountOpt, xCount)

if blockXSize > xCount

    blockSize = blockYSize * blockXSize;

    blockXSize = xCount;
    while (mod(blockSize, blockYSizeAlign * blockXSize) ~= 0) || (blockSize / blockXSize > 256)
        blockXSize = blockXSize + 1;
    end

    blockYSize = blockSize / blockXSize;

elseif blockYSize > yCountOpt

    blockSize = blockYSize * blockXSize;

    blockYSize = ceil(yCountOpt / blockYSizeAlign) * blockYSizeAlign;
    while (mod(blockSize, blockYSize) ~= 0) || (blockSize / blockYSize > 256)
        blockYSize = blockYSize + blockYSizeAlign;
    end

end

end

%***********************************************************************************************************************

function tag = TagOff(rel, seg)

if seg == 0
    tag = rel;
else
    tag = -(seg * 1e10 + rel);
end

end

%***********************************************************************************************************************

function [rel, seg] = RelOff(tag)

if tag < 0
    seg = floor(-tag / 1e10);
    rel = -tag - seg * 1e10;
else
    seg = 0;
    rel = tag;
end

end

%***********************************************************************************************************************

function a = Tag2Abs(a, offs)

if isnumeric(a)

    inds = find(a < 0);
    for i = 1 : numel(inds)
        j = inds(i);
        [rel, seg] = RelOff(a(j));
        a(j) = offs(seg) + rel;
    end

else

    names = fieldnames(a);
    for i = 1 : numel(a)
        for j = 1 : numel(names)
            b = a(i).(names{j});
            if (isnumeric(b) && any(b(:) < 0)) || isstruct(b)
                a(i).(names{j}) = Tag2Abs(b, offs);
            end
        end
    end

end

end

%***********************************************************************************************************************

function [a, vc] = GetField(m, field, t, d)

% Returns a fully padded (siz4b) result (or a scalar).

if isfield(m, field)
    a = m.(field);
    if ~isnumeric(a)
        error('field "%s" must be numeric', field);
    end
elseif ~isequalwithequalnans(d.value, NaN)
    a = d.value;
else
    error('field "%s" is missing', field);
end

a = Internal(a, d.int);

if isempty(t)

    vc = numel(a);

    if d.multi
        if (ndims(a) ~= 2) || all(size(a) > 1)
            error('field "%s" must be a scalar or vector', field);
        end
        a = a(:);
    else
        if ~isscalar(a)
            error('field "%s" must be a scalar', field);
        end
    end

else

    if d.multi
        vc = min(size(a, 1), numel(a));
        t = cns_trans('add', t, vc);
    else
        vc = 1;
    end

    if isempty(a) && (ndims(a) == 2)
        if ~d.multi
            error('field "%s" cannot be empty', field);
        end
        a = reshape(a, t.siz4b);
    elseif isscalar(a)
    elseif cns_trans('sizeis', t, a)
        a = cns_trans('pack', t, a, true, 0); % TODO: roll may not be zero?
    else
        if d.multi
            error('field "%s" must be a scalar, empty, or have size %s', ...
                field, cns_trans('disp', t, true));
        else
            error('field "%s" must be a scalar or have size %s', ...
                field, cns_trans('disp', t));
        end
    end

end

if vc > 65535
    error('field "%s" has more than %u values', field, 65535);
end

end

%***********************************************************************************************************************

function [as, siz2ps] = GetArray(m, field, d, align)

% If align is nonempty, returns fully padded (siz4b) results.  Otherwise results are completely unaligned.

if isfield(m, field)
    as = m.(field);
    if d.multi
        if ~iscell(as), error('"%s" must be a cell array', field); end
        if ~all(cellfun(@isnumeric, as)), error('elements of "%s" must be numeric', field); end
        as = as(:);
    else
        if ~isnumeric(as), error('"%s" must be numeric', field); end
        as = {as};
    end
else
    error('field "%s" is missing', field);
end

for i = 1 : numel(as)
    as{i} = Internal(as{i}, d.int);
end

if isfield(m, [field '_size'])
    sizes = m.([field '_size'])(:)';
    if d.multi
        if ~iscell(sizes), error('"%s_size" must be a cell array', field); end
        if numel(sizes) ~= numel(as)
            error('"%s_size" must have the same number of elements as "%s"', field, field);
        end
    else
        sizes = {sizes};
    end
else
    for i = 1 : numel(as)
        sizes{i} = cns_size(as{i}, numel(d.dims));
    end
end

for i = 1 : numel(as)

    try
        t = cns_trans('create', d, sizes{i}, align);
    catch
        error('"%s": %s', field, cns_error);
    end

    if ~cns_trans('sizeis', t, as{i}), error('"%s" does not match its stated size', field); end

    as    {i} = cns_trans('pack' , t, as{i}, true, 0);
    siz2ps{i} = cns_trans('siz2p', t);

    if any(siz2ps{i} > 65535)
        error('"%s" dimension exceeds %u', field, 65535);
    end

end

end

%***********************************************************************************************************************

function CheckPointer(isType, name, zs, ptrTypeNo, zeroOK)

zs = External(zs, -2);

for i = 1 : numel(zs)

    z = zs(i);

    if zeroOK && (z == 0), continue; end

    if (z < 1) || (z > size(isType, 1)) || (mod(z, 1) ~= 0)
        error('field "%s": invalid layer number', name);
    end

    if ~isType(z, ptrTypeNo)
        error('field "%s": layer number %u is the wrong type', name, z);
    end

end

end

%***********************************************************************************************************************

function [State, varargout] = Run(State, sid, totalIters, varargin)

G = GetSession(State, sid);
if isempty(G.obj), error('session not open'); end

if nargin < 3
    totalIters = 1;
else
    if ~isnumeric(totalIters), error('invalid number of iterations'); end
    totalIters = round(totalIters);
    if totalIters < 0, error('invalid number of iterations'); end
end

args = varargin;

if ~isempty(args) && isnumeric(args{1})
    sampleRate = args{1};
    args = args(2 : end);
    if sampleRate < 1, error('invalid sample rate'); end
else 
    sampleRate = 1;
end

if ~isempty(args) && isnumeric(args{1})
    limitBuffer = true;
    bufferSize = args{1};
    args = args(2 : end);
    if bufferSize < 1, error('invalid buffer size'); end
else
    limitBuffer = false;
    bufferSize = max(totalIters, 1);
end

if ~isempty(args) && ~iscell(args{1})
    error('"get" parameters must be given as cell arrays');
end

[ans, p] = GetAllParams(G, 'r', args{:});

nout = nargout - 1;

if nout == 0

    if ~isempty(p)
        if isequal(sid, -1), error('%u outputs needed', numel(p)); end
        if limitBuffer, error('cannot specify bufferSize when not returning results immediately'); end
    end

    G.p    = p;
    G.time = true;
    State = SaveSession(State, G);

    G.func(4, G.obj, totalIters, sampleRate, 0, 0, p);

    if isequal(sid, -1), G.func(5, G.obj); end

else

    if nout ~= numel(p), error('%u outputs needed', numel(p)); end

    for i = 1 : max(ceil(totalIters / (sampleRate * bufferSize)), 1)

        iters = min(sampleRate * bufferSize, totalIters - (i - 1) * sampleRate * bufferSize);

        G.func(4, G.obj, iters, sampleRate, 0, 0, p);

        [outs{1 : nout}] = G.func(5, G.obj);

        for j = 1 : numel(p)
            outs{j} = GetFinalize(p(j), outs{j}, true);
        end

        if i == 1
            varargout = outs;
        else
            for j = 1 : numel(p)
                varargout{j} = cat(1, varargout{j}, outs{j});
            end
        end

    end

end

end

%***********************************************************************************************************************

function [State, varargout] = Step(State, sid, step1, varargin)

G = GetSession(State, sid);
if isempty(G.obj), error('session not open'); end

if (step1 < 1) || (mod(step1, 1) ~= 0), error('invalid step1'); end

args = varargin;

if ~isempty(args) && isnumeric(args{1})
    step2 = args{1};
    args = args(2 : end);
    if (step2 < step1 - 1) || (mod(step2, 1) ~= 0), error('invalid step2'); end
else 
    step2 = step1;
end

if ~isempty(args) && ~iscell(args{1})
    error('"get" parameters must be given as cell arrays');
end

[ans, p] = GetAllParams(G, 'r', args{:});

nout = nargout - 1;

if nout == 0

    if ~isempty(p)
        if isequal(sid, -1), error('%u outputs needed', numel(p)); end
    end
    
    G.p    = p;
    G.time = false;
    State = SaveSession(State, G);

    G.func(4, G.obj, 1, 1, step1, step2, p);

    if isequal(sid, -1), G.func(5, G.obj); end

else

    if nout ~= numel(p), error('%u outputs needed', numel(p)); end

    G.func(4, G.obj, 1, 1, step1, step2, p);

    [varargout{1 : nout}] = G.func(5, G.obj);

    for i = 1 : numel(p)
        varargout{i} = GetFinalize(p(i), varargout{i}, false);
    end

end

end

%***********************************************************************************************************************

function varargout = Output(State, sid)

if isequal(sid, -1), error('session id required'); end

G = GetSession(State, sid);
if isempty(G.obj), error('session not open'); end

[varargout{1 : nargout}] = G.func(5, G.obj);

% This should always succeed because the C call should fail when there are no outputs.
% This is why we never bother to clear this field.
if ~isfield(G, 'p'), error('no output available'); end

for i = 1 : numel(G.p)
    varargout{i} = GetFinalize(G.p(i), varargout{i}, G.time);
end

end

%***********************************************************************************************************************

function varargout = Get(State, sid, varargin)

G = GetSession(State, sid);
if isempty(G.obj), error('session not open'); end

[ans, p] = GetAllParams(G, 'g', varargin{:});

if nargout ~= numel(p), error('%u outputs needed', numel(p)); end

for i = 1 : numel(p)
    if isempty(p(i).varType)
        if isequal(sid, -1), error('invalid layer number'); end
        varargout{i} = G.user.(p(i).pos);
    else
        varargout{i} = G.func(6, G.obj, p(i));
        varargout{i} = GetFinalize(p(i), varargout{i}, false);
    end
end

end

%***********************************************************************************************************************

function m = Update(State, sid, m)

G = GetSession(State, sid);
if isempty(G.obj), error('session not open'); end
if nargin < 3, error('not enough arguments'); end

for z = 1 : numel(G.def.layers)

    d = G.def.layers{z};
    fieldNames = [d.cat.cv.syms, d.cat.nv.syms, d.cat.nw.syms, d.cat.sv.syms];

    for i = 1 : numel(fieldNames)

        [ans, p] = GetParams(G, 'g', z, fieldNames{i});

        res = G.func(6, G.obj, p);

        res = GetFinalize(p, res, false);
        
        m.layers{z}.(fieldNames{i}) = res;

    end

end

end

%***********************************************************************************************************************

function State = Set(State, sid, varargin)

G = GetSession(State, sid);
if isempty(G.obj), error('session not open'); end

[G, p, a] = GetAllParams(G, 's', varargin{:});

for i = 1 : numel(p)
    if isempty(p(i).varType)
        if isequal(sid, -1), error('invalid layer number'); end
        G.user.(p(i).pos) = a{i};
    else
        G.func(7, G.obj, p(i), a{i});
    end
end

State = SaveSession(State, G);

end

%***********************************************************************************************************************

function [G, p, a] = GetAllParams(G, op, varargin)

if ~isempty(varargin) && ~iscell(varargin{1})
    args = {varargin};
else
    args = varargin;
end

p = [];
a = cell(1, numel(args));

for i = 1 : numel(args)

    if ~iscell(args{i}), error('invalid argument'); end

    if i == 1
        [G, p, a{i}] = GetParams(G, op, args{i}{:});
    else
        [G, p(i), a{i}] = GetParams(G, op, args{i}{:});
    end

end

end

%***********************************************************************************************************************

function [G, p, a] = GetParams(G, op, z, varargin)

if nargin < 3, error('incorrect number of arguments'); end

if ischar(z)
    [p, a] = GetHostParams(op, z, varargin{:});
    return;
end

if nargin < 4, error('incorrect number of arguments'); end

fieldName = varargin{1};
args      = varargin(2 : end);

if z == 0
    c = G.s;
    if strcmp(fieldName, 'iter_no')
        d = struct('cat', {'sp'});
    else
        d = G.def.sym.(fieldName);
    end
else
    c = G.s.layers(z);
    d = G.def.layers{z}.sym.(fieldName);
end

if op == 's'
    if isempty(args), error('z=%u, field "%s": incorrect number of arguments', z, fieldName); end
    a = args{end};
    args = args(1 : end - 1);
    if ~isnumeric(a), error('z=%u, field "%s": invalid value', z, fieldName); end
else
    a = [];
end

switch d.cat
case {'mp', 'gp', 'lp'}, [G, p, a] = GetParams_C (G, op, z, fieldName, c, d, a, args{:});
case {'ma', 'ga', 'la'}, [G, p, a] = GetParams_A (G, op, z, fieldName, c, d, a, args{:});
case {'mt', 'gt', 'lt'}, [G, p, a] = GetParams_T (G, op, z, fieldName, c, d, a, args{:});
case 'cc'              , [G, p, a] = GetParams_CC(G, op, z, fieldName, c, d, a, args{:});
case 'cv'              , [G, p, a] = GetParams_CV(G, op, z, fieldName, c, d, a, args{:});
case {'nc', 'nv'}      , [G, p, a] = GetParams_N (G, op, z, fieldName, c, d, a, args{:});
case 'nw'              , [G, p, a] = GetParams_W (G, op, z, fieldName, c, d, a, args{:});
case {'sc', 'sv'}      , [G, p, a] = GetParams_S (G, op, z, fieldName, c, d, a, args{:});
case 'sp'              , [G, p, a] = GetParams_SP(G, op, z, fieldName, c, d, a, args{:});
otherwise
    error('z=%u, field "%s": cannot get/set this field', z, fieldName);
end

if op == 's'
    a = Internal(a, p.int);
    t = p.t;
    if numel(a) == 1
        a = repmat(a, t.siz4b);
    elseif cns_trans('sizeis', t, a)
        a = cns_trans('pack', t, a, true, 0);
    else
        error('z=%u, field "%s": value must be a scalar or have size %s', ...
            z, fieldName, cns_trans('disp', t));
    end
else
    a = single([]);
end

end

%***********************************************************************************************************************

function [p, a] = GetHostParams(op, name, value)

switch op
case 'g'
    if nargin >= 3, error('property "%s": incorrect number of arguments', name); end
    a = [];
case 's'
    if nargin < 3, error('property "%s": incorrect number of arguments', name); end
    a = value;
case 'r'
    error('property "%s" cannot be retrieved in run mode', name);
end

p.varType = [];
p.pos     = name;
p.height  = []; % unused
p.width   = []; % unused
p.hOff    = []; % unused
p.wOff    = []; % unused
p.dOff    = []; % unused
p.hCount  = []; % unused
p.wCount  = []; % unused
p.dCount  = []; % unused
p.t       = []; % unused
p.int     = []; % unused

end

%***********************************************************************************************************************

function [G, p, a] = GetParams_C(G, op, z, fieldName, c, d, a, varargin)

if op == 'r'
    error('z=%u: field "%s" cannot be retrieved in run mode', z, fieldName);
end

[q, foff] = GetCoords(G, z, fieldName, c, d, false, false, varargin{:});

if ~d.multi
    if d.cat(1) == 'g'
        foff = c.gcOff + foff;
    else
        foff = c.cOff + foff;
    end
end

p.varType = 0;
p.pos     = 0; % unused
p.height  = 1; % unused
p.width   = 1; % unused
p.hOff    = q.yo + foff;
p.wOff    = 0;
p.dOff    = 0;
p.hCount  = q.yc;
p.wCount  = 1;
p.dCount  = 1;
p.t       = q.t;
p.int     = d.int;

end

%***********************************************************************************************************************

function [G, p, a] = GetParams_A(G, op, z, fieldName, c, d, a, varargin)

if op == 'r'
    error('z=%u: field "%s" cannot be retrieved in run mode', z, fieldName);
end

q = GetArrayCoords(G, z, fieldName, c, d, G.pr.blockYSizeAlign, varargin{:});

p.varType = 1;
p.pos     = G.s.cMeta(q.eoff - 1) + G.s.cMeta(q.eoff) * 65536;
p.height  = q.at.siz4b(1);
p.width   = q.at.siz4b(2);
p.hOff    = q.yo;
p.wOff    = q.xo;
p.dOff    = 0;
p.hCount  = q.yc;
p.wCount  = q.xc;
p.dCount  = 1;
p.t       = q.t;
p.int     = d.int;

end

%***********************************************************************************************************************

function [G, p, a] = GetParams_T(G, op, z, fieldName, c, d, a, varargin)

if op == 'r'
    error('z=%u: field "%s" cannot be retrieved in run mode', z, fieldName);
end

q = GetArrayCoords(G, z, fieldName, c, d, [], varargin{:});

p.varType = 2;
p.pos     = d.resNo - 1;
p.height  = 1; % unused
p.width   = 1; % unused
p.hOff    = q.yo + G.s.cMeta(q.eoff - 1);
p.wOff    = q.xo + G.s.cMeta(q.eoff);
p.dOff    = 0;
p.hCount  = q.yc;
p.wCount  = q.xc;
p.dCount  = 1;
p.t       = q.t;
p.int     = d.int;

end

%***********************************************************************************************************************

function [G, p, a] = GetParams_CC(G, op, z, fieldName, c, d, a, varargin)

if op == 'r'
    error('z=%u: field "%s" cannot be retrieved in run mode', z, fieldName);
end

q = GetCoords(G, z, fieldName, c, d, false, true, varargin{:});

p.varType = 3;
p.pos     = d.resNo - 1;
p.height  = 1; % unused
p.width   = 1; % unused
p.hOff    = q.yo + G.s.cMeta(c.tOff + 2 * d.pos - 1);
p.wOff    = q.xo + G.s.cMeta(c.tOff + 2 * d.pos);
p.dOff    = 0;
p.hCount  = q.yc;
p.wCount  = q.xc;
p.dCount  = 1;
p.t       = q.t;
p.int     = d.int;

end

%***********************************************************************************************************************

function [G, p, a] = GetParams_CV(G, op, z, fieldName, c, d, a, varargin)

q = GetCoords(G, z, fieldName, c, d, false, true, varargin{:});

p.varType = 4;
p.pos     = d.resNo - 1;
p.height  = G.s.cvYSizes (d.resNo);
p.width   = G.s.cvXCounts(d.resNo);
p.hOff    = q.yo + G.s.cMeta(c.tOff + 2 * d.pos - 1);
p.wOff    = q.xo + G.s.cMeta(c.tOff + 2 * d.pos);
p.dOff    = 0;
p.hCount  = q.yc;
p.wCount  = q.xc;
p.dCount  = 1;
p.t       = q.t;
p.int     = d.int;

end

%***********************************************************************************************************************

function [G, p, a] = GetParams_N(G, op, z, fieldName, c, d, a, varargin)

if (op == 'r') && strcmp(d.cat, 'nc')
    error('z=%u: field "%s" cannot be retrieved in run mode', z, fieldName);
end

[q, foff] = GetCoords(G, z, fieldName, c, d, false, true, varargin{:});

p.varType = 5;
p.pos     = z - 1;
p.height  = c.ySize;
p.width   = c.xCount;
p.hOff    = q.yo;
p.wOff    = q.xo;
p.dOff    = q.fo + foff;
p.hCount  = q.yc;
p.wCount  = q.xc;
p.dCount  = q.fc;
p.t       = q.t;
p.int     = d.int;

end

%***********************************************************************************************************************

function [G, p, a] = GetParams_W(G, op, z, fieldName, c, d, a, varargin)

[q, foff] = GetCoords(G, z, fieldName, c, d, false, true, varargin{:});

p.varType = 6;
p.pos     = z - 1;
p.height  = c.ySize;
p.width   = c.xCount;
p.hOff    = q.yo;
p.wOff    = q.xo;
p.dOff    = q.fo + foff;
p.hCount  = q.yc;
p.wCount  = q.xc;
p.dCount  = q.fc;
p.t       = q.t;
p.int     = d.int;

end

%***********************************************************************************************************************

function [G, p, a] = GetParams_S(G, op, z, fieldName, c, d, a, varargin)

if (op == 'r') && strcmp(d.cat, 'sc')
    error('z=%u: field "%s" cannot be retrieved in run mode', z, fieldName);
end

[q, foff] = GetCoords(G, z, fieldName, c, d, true, true, varargin{:});

p.varType = 7;
p.pos     = z - 1;
p.height  = c.ySize;
p.width   = c.xCount;
p.hOff    = q.yo;
p.wOff    = q.xo;
p.dOff    = q.fo + foff;
p.hCount  = q.yc;
p.wCount  = q.xc;
p.dCount  = q.fc;
p.t       = q.t;
p.int     = d.int;

end

%***********************************************************************************************************************

function [G, p, a] = GetParams_SP(G, op, z, fieldName, c, d, a)

% Currently only used for "iter_no".

switch op
case 's'
    if ~isscalar(a) || (a < 1) || (a > cns_intmax) || (mod(a, 1) ~= 0)
        error('z=%u, field "%s": invalid value', z, fieldName);
    end
case 'r'
    error('z=%u: field "%s" cannot be retrieved in run mode', z, fieldName);
end

p.varType = 8;
p.pos     = 0; % unused
p.height  = 1; % unused
p.width   = 1; % unused
p.hOff    = 0;
p.wOff    = 0;
p.dOff    = 0;
p.hCount  = 1;
p.wCount  = 1;
p.dCount  = 1;
p.t       = cns_trans('scalar');
p.int     = -2;

end

%***********************************************************************************************************************

function [q, foff] = GetCoords(G, z, fieldName, c, d, hasSyns, hasYX, varargin)

counts = [];

if d.multi
    if d.cat(1) == 'g'
        pos = c.gmvOff + 2 * d.pos - 1;
    else
        pos = c.mvOff + 2 * d.pos - 1;
    end
    foff = G.s.cMeta(pos);
    counts(end + 1) = G.s.cMeta(pos + 1);
else    
    foff = d.pos - 1;
end

if hasSyns
    foff = foff * c.sSize;
    counts(end + 1) = c.sSize;
end

varargin(end + 1 : numel(counts)) = {[]};

if hasYX
    t     = c.t;
    shift = G.s.sh.shift(z);
    roll  = G.s.sh.roll (z);
    if numel(varargin) == numel(counts), varargin(end + 1 : numel(counts) + numel(t.siz1)) = {[]}; end
else
    t     = cns_trans('scalar');
    shift = 0;
    roll  = 0;
end

if ~isempty(counts)
    t = cns_trans('add', t, counts);
end

q = cns_trans('range', t, shift, roll, varargin{:});

end

%***********************************************************************************************************************

function q = GetArrayCoords(G, z, fieldName, c, d, align, varargin)

if d.cat(1) == 'g'
    pos = c.gmvOff + 2 * d.pos - 1;
else
    pos = c.mvOff  + 2 * d.pos - 1;
end
eoff = G.s.cMeta(pos);
num  = G.s.cMeta(pos + 1);

if d.multi
    if isempty(varargin)
        error('z=%u, field "%s": missing value number', z, fieldName);
    end
    e = GetCoordValue(varargin{1}, num);
    if isempty(e)
        error('z=%u, field "%s": invalid value number', z, fieldName);
    end
    varargin = varargin(2 : end);
    fo = e - 1;
else
    fo = 0;
end

eoff = eoff + fo * ceil((2 + sum(cellfun(@numel, d.dims)) + 2) / 2) * 2;

siz2 = G.s.cMeta(eoff + 1 : eoff + sum(cellfun(@numel, d.dims)));
at = cns_trans('recreate', d, siz2, align);

if isempty(varargin), varargin(1 : numel(at.siz1)) = {[]}; end

q = cns_trans('range', at, 0, 0, varargin{:});

q.eoff = eoff;
q.at   = at;

end

%***********************************************************************************************************************

function e = GetCoordValue(a, n)

e = [];

if ~isnumeric(a) || ~isscalar(a), return; end
a = double(a);
if mod(a, 1) ~= 0, return; end

if (a < 1) || (a > n), return; end
e = a;

end

%***********************************************************************************************************************

function res = GetFinalize(p, res, time)

t = p.t;

if time, t = cns_trans('add', t, size(res, 2)); end

res = cns_trans('unpack', t, reshape(res, t.siz4b), 0);

res = External(res, p.int);

end

%***********************************************************************************************************************

function [State, m] = Shift(State, sid, inc, m)

% TODO: should prevent t_current, t_start, etc. parameters ('mp' or 'lp' fields) from being set directly.

G = GetSession(State, sid);
if isempty(G.obj), error('session not open'); end

if ~isscalar(inc) || (inc < 0), error('invalid shift value'); end
if (nargin < 4) ~= (nargout < 2), error('invalid shift parameters'); end

sn = G.s.sn;
sh = G.s.sh;
if isempty(sn), error('no shiftable dimension'); end

sh.current = sh.current + inc;

sh.shift = sh.flag .* max(1 + floor((sh.current - sh.lag - sh.start) ./ sh.space + 0.001) - sh.size, 0);
z = find(sh.shift > sh.maxShift, 1);
if ~isempty(z)
    error('z=%u: maximum shift is %u, %u attempted', z, sh.maxShift(z), sh.shift(z));
end

sh.start = sh.start - sh.total .* sh.space;
sh.total = sh.total + sh.shift;
sh.start = sh.start + sh.total .* sh.space;

sh.roll = ComputeRoll(sh.total, sh.size);

G.s.sh = sh;

G.func(8, G.obj, sh);

State = SaveSession(State, G);

if nargin >= 4
    m.([sn '_current']) = sh.current;
    for z = 1 : numel(sh.flag)
        if ~sh.flag(z), continue; end
        m.layers{z}.([sn '_start']) = sh.start(z);
        m.layers{z}.([sn '_total']) = sh.total(z);
        m.layers{z}.([sn '_shift']) = sh.shift(z);
    end
end

end

%***********************************************************************************************************************

function roll = ComputeRoll(total, siz)

roll = mod(siz - mod(total, siz), siz);

end

%***********************************************************************************************************************

function sid = Wait(State, sids, varargin)

sid = Wait2(State, sids, true, varargin{:});

end

%***********************************************************************************************************************

function sid = Poll(State, sids, varargin)

sid = Wait2(State, sids, false, varargin{:});

end

%***********************************************************************************************************************

function sid = Wait2(State, sids, wait, flag)

if nargin < 4
    if isequal(sids, -1), error('session id(s) required'); end
else
    if ~isequal(sids, -1), error('too many arguments'); end
    if ~strcmp(flag, 'any'), error('invalid parameter'); end
    sids = AllSessions(State);
end

funcs = cell (1, numel(sids));
objs  = zeros(1, numel(sids), 'uint64');

for i = 1 : numel(sids)
    if sids(i) == 0, error('invalid session id'); end
    G = GetSession(State, sids(i));
    if isempty(G.obj), error('session %u not open', G.sid); end
    sids (i) = G.sid;
    funcs{i} = G.func;
    objs (i) = G.obj;
end

if isempty(sids)

    if ~wait, error('no sessions to poll'); end

    sid = [];

elseif isscalar(sids) && wait

    funcs{1}(9, objs(1));

    sid = sids;

else

    % For now we just loop and poll.

    sid = [];

    while true

        for i = 1 : numel(sids)
            if funcs{i}(10, objs(i)) == 0
                sid = sids(i);
                break;
            end
        end

        if ~isempty(sid) || ~wait, break; end

        pause(0.01);

    end

end

sid = uint32(sid);

end

%***********************************************************************************************************************

function sids = Sessions(State, sid)

if ~isequal(sid, -1), error('session id not allowed'); end

sids = uint32(AllSessions(State));

end

%***********************************************************************************************************************

function count = DevCount(State, sid, platform)

if ~isequal(sid, -1), error('session id not allowed'); end

if nargin < 3
    args = State.args;
else
    args = {platform};
end

p = GetPlatform(args{:});

if strcmp(p.platform, 'cuda')
    count = cns_devcount_cuda;
    if (count > 0) && (~State.multi || (p.deviceNo >= 0)), count = 1; end
else
    count = inf;
end

end

%***********************************************************************************************************************

function a = Internal(a, int)

if int < 0
    if int == -2, a = a - 1; end
    a = cns_intin(int32(a));
else
    a = single(a);
end

end

%***********************************************************************************************************************

function a = External(a, int)

if int < 0
    a = cns_intout(a);
    if int == -2, a = a + 1; end
end

end

%***********************************************************************************************************************

function cb = CB(G)

cb = @Callback;

function varargout = Callback(cbname, varargin)

cbfunc = str2func(cbname);
[varargout{1 : nargout}] = cbfunc(G, varargin{:});

end

end

%***********************************************************************************************************************

function s = CBYX2S(G, z, y, x, roll, internal)

c = G.s.layers(z);
if iscell(c), c = c{1}; end

e = cns_trans('yx2e', c.t, y, x, roll);

% TODO: should probably use split dimensions if internal

names = G.def.layers{z}.dnames;
if internal, names = upper(names); end

[coords{1 : numel(names)}] = cns_iconv(G.m, z, e);
if internal, coords = num2cell([coords{:}] - 1); end

args = [names; coords];
s = sprintf('%s=%u ', args{:});
s = s(1 : end - 1);

end

%***********************************************************************************************************************

function G = GetSession(State, sid)

if ~isscalar(sid), error('invalid session id'); end

switch sid
case -1
    sid = 1;
case 0
    for i = 1 : numel(State.gs)
        if isempty(State.gs{i})
            sid = i;
            break;
        end
    end
    if sid == 0, sid = numel(State.gs) + 1; end
end

if (sid > numel(State.gs)) || isempty(State.gs{sid})
    G.sid = sid;
    G.obj = [];
else
    G = State.gs{sid};
end

end

%***********************************************************************************************************************

function State = SaveSession(State, G)

if isempty(G.obj)

    State.gs{G.sid} = [];

    while ~isempty(State.gs) && isempty(State.gs{end})
        State.gs = State.gs(1 : end - 1);
    end

else

    State.gs{G.sid} = G;

end

end

%***********************************************************************************************************************

function sids = AllSessions(State)

sids = [];

for i = 1 : numel(State.gs)
    if ~isempty(State.gs{i})
        sids(end + 1) = i;
    end
end

end
