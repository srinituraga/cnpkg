function def = cns_def(m, varargin)

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

if ischar(m)
    def = GetDef(m, varargin{:});
else
    def = GetModelDef(m, varargin{:});
end

return;

%***********************************************************************************************************************

function def = GetModelDef(m, varargin)

d = GetDef(m.package, varargin{:});

def = rmfield(d, {'types', 'type'});

if isfield(m, 'layers'), numLayers = numel(m.layers); else numLayers = 0; end
if isfield(m, 'groups'), numGroups = numel(m.groups); else numGroups = 0; end

def.layers = cell(1, numLayers);

gs = zeros(1, numLayers);
ts = zeros(1, numLayers);

for z = 1 : numLayers

    name = m.layers{z}.type;
    if ~isfield(d.type, name)
        error('z=%u: type name "%s" is invalid', z, name);
    end
    if d.type.(name).abstract
        error('z=%u: type "%s" is abstract', z, name);
    end

    def.layers{z} = d.type.(name);
    def.layers{z}.type = name;

    if isfield(m.layers{z}, 'groupNo'), gs(z) = m.layers{z}.groupNo; end
    ts(z) = def.layers{z}.typeNo;

end

gCount = max([gs 0]);
if numel(unique(gs(gs > 0))) ~= gCount
    error('group numbers must be contiguous');
end
if numGroups ~= gCount
    error('"groups" must have %u elements', gCount);
end

def.groups = cell(1, gCount + sum(gs == 0));
def.gCount = gCount;

for g = 1 : gCount

    zs = find(gs == g);
    if ~isscalar(unique(ts(zs))), error('all layers in a group must be of the same type'); end

    def.groups{g} = struct;
    def.groups{g}.zs = zs;

    for z = zs
        def.layers{z}.g    = g;
        def.layers{z}.auto = false;
    end

end

g = gCount;

for z = find(gs == 0)

    g = g + 1;

    def.groups{g} = struct;
    def.groups{g}.zs = z;

    def.layers{z}.g    = g;
    def.layers{z}.auto = true;

end

return;

%***********************************************************************************************************************

function def = GetDef(package, rebuild)

if nargin < 2, rebuild = false; end

persistent Cache;
if isempty(Cache), Cache = struct; end

if ~rebuild && isfield(Cache, package)
    def = Cache.(package);
    return;
end

func = [package '_cns'];
path = fileparts(which(func));
if isempty(path), error('cannot find definition function "%s.m"', func); end
saveFile = fullfile(path, [func '_compiled_def.mat']);

if ~rebuild && exist(saveFile, 'file')
    load(saveFile, 'def');
else
    def = BuildDef(path, func);
    save(saveFile, 'def');
end

Cache.(package) = def;

return;

%***********************************************************************************************************************

function def = BuildDef(path, func)

h = Init;

[h, def, udef, udefs] = ReadMFiles(h, path, func);

h.def = def; % Save the definition: post-properties, pre-fields.

[def, h] = InitDef(def, h, [], {});
[def, h] = InitFields(def, h, 'mp', 'mp', 'mvm', false, 2, '' );
[def, h] = InitFields(def, h, 'mz', 'mp', 'mvm', false, 1, '' );
[def, h] = InitFields(def, h, 'ma', ''  , 'mvm', false, 0, 'a');
[def, h] = InitFields(def, h, 'mt', ''  , 'mvm', false, 0, 't');
[def, h] = InitConsts(def, h);
[def, h] = InitExtras(def, h);

try
    [def, h] = Parse(def, h, udef, 'global', 0);
catch
    error('%s.m: %s', func, cns_error);
end

for i = 1 : numel(def.types)

    name = def.types{i};
    d = def.type.(name);

    [d, h] = InitDef(d, h, def.refTypeNos, def.syms);
    [d, h] = InitFields(d, h, 'gp', 'gp', 'mvg', true , 2, ''  );
    [d, h] = InitFields(d, h, 'gz', 'gp', 'mvg', true , 1, ''  );
    [d, h] = InitFields(d, h, 'ga', ''  , 'mvg', true , 0, 'a' );
    [d, h] = InitFields(d, h, 'gt', ''  , 'mvg', true , 0, 't' );
    [d, h] = InitFields(d, h, 'lp', 'lp', 'mvl', false, 2, ''  );
    [d, h] = InitFields(d, h, 'lz', 'lp', 'mvl', false, 1, ''  );
    [d, h] = InitFields(d, h, 'la', ''  , 'mvl', false, 0, 'a' );
    [d, h] = InitFields(d, h, 'lt', ''  , 'mvl', false, 0, 't' );
    [d, h] = InitFields(d, h, 'cc', 'c' , 'mvl', false, 1, 'cc');
    [d, h] = InitFields(d, h, 'cv', 'c' , 'mvl', false, 1, 'cv');
    [d, h] = InitFields(d, h, 'nc', 'n' , 'mvl', false, 1, 'nc');
    [d, h] = InitFields(d, h, 'nv', 'n' , 'mvl', false, 1, ''  );
    [d, h] = InitFields(d, h, 'nw', ''  , 'mvl', false, 1, 'nw');
    [d, h] = InitFields(d, h, 'sc', 's' , 'mvl', false, 1, ''  );
    [d, h] = InitFields(d, h, 'sv', 's' , 'mvl', false, 1, ''  );
    [d, h] = InitConsts(d, h);
    [d, h] = InitArrays(d, h);
    [d, h] = InitExtras(d, h);

    for j = 1 : numel(d.typePath)

        [ans, k] = ismember(d.typePath{j}, def.types);

        try
            [d, h] = Parse(d, h, udefs{k}, def.types{k}, k);
        catch
            error('%s_type_%s.m: %s', func, def.types{k}, cns_error);
        end

    end

    def.type.(name) = d;

end

def = Done(def, h);

return;

%***********************************************************************************************************************

function [h, def, udef, udefs] = ReadMFiles(h, path, func)

def = struct;

% Process global definitions.

uprop = feval(func, 'props' );
udef  = feval(func, 'fields');

def.typeNo = 0;
def.refTypeNos = [];

if isfield(uprop, 'methods')
    def.methods = uprop.methods;
else
    def.methods = {};
end

% Process type definitions.

list = dir(fullfile(path, [func '_type_*.m']));

names  = cell(1, numel(list));
s      = struct;
tfuncs = cell(1, numel(list));
uprops = cell(1, numel(list));
udefs  = cell(1, numel(list));

for i = 1 : numel(list)

    tfuncs{i} = list(i).name(1 : end - 2);
    names{i} = tfuncs{i}(numel(func) + 7 : end);

    if any(names{i} == '_')
        error('%s.m: type names cannot contain underscores', tfuncs{i});
    end
    if any(strcmpi(names{i}, {'global'}))
        error('%s.m: invalid type name', tfuncs{i});
    end

    uprops{i} = feval(tfuncs{i}, 'props' );
    udefs {i} = feval(tfuncs{i}, 'fields');

    d = struct;

    if isfield(uprops{i}, 'super')
        d.super = uprops{i}.super;
    else
        d.super = '';
    end

    if strcmp(names{i}, 'base')
        if ~isempty(d.super), error('%s.m: base type cannot have a superclass', tfuncs{i}); end
    else
        if isempty(d.super), d.super = 'base'; end
    end
    
    s.(names{i}) = d;

end

if ~ismember('base', names), 
    error('a "base" type is required');
end

keys = cell(1, numel(names));

for i = 1 : numel(names)

    tpath = {};

    name = names{i};
    while true
        tpath = [{name}, tpath];
        name = s.(name).super;
        if isempty(name), break; end
        if ~ismember(name, names)
            error('%s_type_%s.m: superclass "%s" not found', func, tpath{1}, name);
        end
        if ismember(name, tpath)
            error('%s_type_%s.m: superclasses form a loop', func, tpath{1});
        end
    end

    s.(names{i}).typePath = tpath;

    keys{i} = strcat(tpath, '*');
    keys{i} = [keys{i}{:}];

end

[keys, inds] = sort(keys);
names  = names(inds);
s      = orderfields(s, inds);
tfuncs = tfuncs(inds);
uprops = uprops(inds);
udefs  = udefs (inds);

for i = 1 : numel(names)

    d = s.(names{i});

    d.typeNo = i;

    [ans, allTypeNos] = ismember(d.typePath, names);
    d.isType = false(1, numel(names));
    d.isType(allTypeNos) = true;

    d.refTypeNos = i;

    if isfield(uprops{i}, 'abstract'), d.abstract = uprops{i}.abstract; else d.abstract = false      ; end
    if isfield(uprops{i}, 'kernel'  ), d.kernel   = uprops{i}.kernel  ; else d.kernel   = ~d.abstract; end

    if d.abstract && d.kernel
        error('%s.m: abstract types cannot have kernels', tfuncs{i});
    end   

    if d.kernel
        d.blockYSize = uprops{i}.blockYSize;
        d.blockXSize = uprops{i}.blockXSize;
        if any([d.blockYSize, d.blockXSize] > 256)
            error('%s.m: blockYSize and blockXSize cannot exceed 256', tfuncs{i});
        end
    else
        if any(isfield(uprops{i}, {'blockYSize', 'blockXSize'}))
            error('%s.m: cannot specify block sizes if there is no kernel', tfuncs{i});
        end
    end

    d.methods = {};
    d.method  = struct;

    d.dimNo     = 0;
    d.dimTypeNo = 0;
    d = cns_trans('undef', d);

    for j = 1 : numel(d.typePath)

        [ans, k] = ismember(d.typePath{j}, names);

        if isfield(uprops{k}, 'methods')
            methods = uprops{k}.methods;
        else
            methods = {};
        end
        for n = 1 : numel(methods)
            if ismember(methods{n}, d.methods)
                d.method.(methods{n})(end + 1) = names(k);
            else
                d.methods(end + 1) = methods(n);
                d.method.(methods{n}) = names(k);
            end
        end

        if any(isfield(uprops{k}, {'dims', 'dparts', 'dnames', 'dmap'}))
            if d.dimNo ~= 0, error('%s.m: cannot redefine dims', tfuncs{k}); end
            try
                d = cns_trans('parse', d, uprops{k}, true);
            catch
                error('%s.m: %s', tfuncs{k}, cns_error);
            end
            [h, d.dimNo] = GetResNo(h, 'dim', sprintf('%u', k));
            d.dimTypeNo = k;
            h.maxSiz2p = max(h.maxSiz2p, sum(cellfun(@numel, d.dims)) + 2 + sum(d.dmap == 2));
        end

    end

    if ~d.abstract && (d.dimNo == 0)
        error('%s.m: non-abstract types must define dims', tfuncs{i});
    end

    s.(names{i}) = d;

end

for i = 1 : numel(names)

    d = s.(names{i});

    d.synTypeNo = 0;

    for j = 1 : numel(d.typePath)

        [ans, k] = ismember(d.typePath{j}, names);

        if isfield(uprops{k}, 'synType')
            if d.synTypeNo ~= 0, error('%s.m: cannot redefine synType', tfuncs{k}); end
            [ans, d.synTypeNo] = ismember(uprops{k}.synType, names);
            if d.synTypeNo == 0, error('%s.m: invalid synType', tfuncs{k}); end
            if s.(uprops{k}.synType).dimNo == 0
                error('%s.m: synType "%s" does not define dims', tfuncs{k}, uprops{k}.synType);
            end
            % TODO: synType must not have any shiftable dimensions
            d.refTypeNos = union(d.refTypeNos, d.synTypeNo);
        end

    end

    s.(names{i}) = d;

    if d.dimTypeNo == i
        for j = 1 : numel(d.dims)
            if d.dmap(j)
                udefs{i} = AddToDef(tfuncs{i}, udefs{i}, [d.dnames{j} '_start'], 'lp');
                udefs{i} = AddToDef(tfuncs{i}, udefs{i}, [d.dnames{j} '_space'], 'lp');
            end
            if d.dmap(j) == 2
                if isempty(h.shiftName)
                    h.shiftName = d.dnames{j};
                    udef = AddToDef(func, udef, [d.dnames{j} '_current'], 'mp');
                elseif ~strcmp(d.dnames{j}, h.shiftName)
                    error('%s.m: shiftable dimension name must be the same for all types', tfuncs{i});
                end
                % TODO: option to make these 'lp' fields?
                udefs{i} = AddToDef(tfuncs{i}, udefs{i}, [d.dnames{j} '_lag'     ], 'x');
                udefs{i} = AddToDef(tfuncs{i}, udefs{i}, [d.dnames{j} '_maxShift'], 'x');
                udefs{i} = AddToDef(tfuncs{i}, udefs{i}, [d.dnames{j} '_total'   ], 'x');
                udefs{i} = AddToDef(tfuncs{i}, udefs{i}, [d.dnames{j} '_shift'   ], 'lp', 'int', 'dflt', 0);
            end
        end
    end

end

def.types = names;
def.type  = s;

return;

%***********************************************************************************************************************

function udef = AddToDef(func, udef, name, cat, varargin)

if isfield(udef, name), error('%s.m: "%s" is already defined', func, name); end

udef.(name) = {cat, varargin{:}};

return;

%***********************************************************************************************************************

function [d, h] = Parse(d, h, u, typeName, typeNo)

names = fieldnames(u);

for i = 1 : numel(names)

    name = names{i};

    [cat, p] = GetSymbol(u, name);

    if ~ismember(cat, d.cats) && ~strcmp(cat, 'd')
        error('symbol "%s": invalid definition', name);
    end

    p.typeName = typeName;
    p.typeNo   = typeNo;

    if strcmp(cat, 'c')

        [d, h] = AddConst(d, h, name, p);

    elseif strcmp(cat, 'd')

        [d, h, p] = SetDflt(d, h, name(1 : end - 5), p);
        [d, h] = AddConst(d, h, name, p);

    elseif strcmp(cat, 'a')

        [d, h] = AddArray(d, h, name, p);

    elseif strcmp(cat, 'x')

        [d, h] = AddExtra(d, h, name, p);

    else

        [d, h] = AddField(d, h, name, cat, p);

        if ~isequalwithequalnans(p.value, NaN)
            [d, h] = SetDflt(d, h, name, p);
            [d, h] = AddConst(d, h, [name '_dflt'], p);
        end

    end

end

return;

%***********************************************************************************************************************

function [cat, p] = GetSymbol(u, name)

cat = u.(name);
if ischar(cat)
    rest = {};
elseif iscell(cat) && ~isempty(cat) && ischar(cat{1})
    rest = cat(2 : end);
    cat = cat{1};
elseif isnumeric(cat)
    rest = {cat};
    cat = 'c';
elseif iscell(cat) && ~isempty(cat) && isnumeric(cat{1})
    rest = cat;
    cat = 'c';
else
    error('symbol "%s": invalid definition', name);
end

if strcmp(cat, 'd')
    error('symbol "%s": invalid definition', name);
end
if ~isempty(regexp(name, '_dflt$', 'once'))
    if ~strcmp(cat, 'c')
        error('symbol name "%s" is invalid for this category', name);
    end
    cat = 'd';
end

switch cat
case {'c', 'd'}, p = GetConstSymbol(name, rest);
case 'a'       , p = GetArraySymbol(name, rest);
case 'x'       , p = GetExtraSymbol(name, rest);
otherwise      , p = GetFieldSymbol(name, rest);
end

return;

%***********************************************************************************************************************

function h = Init

h = struct;

h.badNames = cns_reservednames;

h.dimList   = {};
h.maxSiz2p  = 0;
h.shiftName = '';

h.aList  = {};
h.tList  = {};
h.ccList = {};
h.cvList = {};
h.ncList = {};
h.nwList = {};

return;

%***********************************************************************************************************************

function [d, h] = InitDef(d, h, refTypeNos, globalNames)

d.refTypeNos = union(d.refTypeNos, refTypeNos);

d.cats  = {};
d.cat   = struct;
d.lists = {};
d.list  = struct;
d.syms  = {};
d.sym   = struct;

h.globalNames = globalNames;

return;

%***********************************************************************************************************************

function [d, h] = InitFields(d, h, cat, list, mvList, group, dfltOK, res)

c = struct;

c.field  = true;
c.list   = list;
c.mvList = mvList;
c.group  = group;
c.syms   = {};
c.svSyms = {};
c.mvSyms = {};

d.cats{end + 1} = cat;
d.cat.(cat)     = c;

if ~isempty(list) && ~ismember(list, d.lists)
    d.lists{end + 1} = list;
    d.list.(list).syms   = {};
    d.list.(list).svSyms = {};
    d.list.(list).mvSyms = {};
end

if ~ismember(mvList, d.lists)
    d.lists{end + 1} = mvList;
    d.list.(mvList).syms = {};
end

h.(cat)        = struct;
h.(cat).dfltOK = dfltOK;
h.(cat).res    = res;

return;

%***********************************************************************************************************************

function p = GetFieldSymbol(name, rest)

p.multi   = false;
p.int     = 0;
p.value   = NaN;
p.private = false;
p.cache   = false;

used = {};

while ~isempty(rest)

    if ~ischar(rest{1}) || ismember(rest{1}, used)
        error('field "%s": invalid definition', name);
    end
    used{end + 1} = rest{1};

    if strcmp(rest{1}, 'mv')

        p.multi = true;
        rest = rest(2 : end);

    elseif strcmp(rest{1}, 'int')

        p.int = -1;
        rest = rest(2 : end);
        used{end + 1} = 'ind';

    elseif strcmp(rest{1}, 'ind')

        p.int = -2;
        rest = rest(2 : end);
        used{end + 1} = 'int';

    elseif strcmp(rest{1}, 'dflt') && (numel(rest) >= 2) && isnumeric(rest{2}) && ~isequalwithequalnans(rest{2}, NaN)

        p.value = rest{2};
        rest = rest(3 : end);

    elseif ismember(rest{1}, {'dims', 'dparts', 'dnames'}) && (numel(rest) >= 2) && iscell(rest{2})

        p.(rest{1}) = rest{2};
        rest = rest(3 : end);

    elseif strcmp(rest{1}, 'type') && (numel(rest) >= 2) && ischar(rest{2})

        p.ptrType = rest{2};
        rest = rest(3 : end);

    elseif strcmp(rest{1}, 'private')

        p.private = true;
        rest = rest(2 : end);

    elseif strcmp(rest{1}, 'cache')

        p.cache = true;
        rest = rest(2 : end);

    else

        error('field "%s": invalid definition', name);

    end

end

return;

%***********************************************************************************************************************

function [d, h] = AddField(d, h, name, cat, p)

CheckName(d, h, name);

if (cat(1) == 'm') || (cat(2) == 't') || ismember(cat, {'nw'})
    if p.private, error('field "%s": "private" is invalid for this category', name); end
elseif cat(1) == 'n'
    if p.private, error('field "%s": "private" is invalid for this category', name); end
    p.private = true;
elseif (cat(2) == 'z') || (cat(1) == 's')
    p.private = true;
end

if (cat(2) ~= 'a') && (cat(1) ~= 'c')
    if p.cache, error('field "%s": "cache" is invalid for this category', name); end
end

if ismember(cat, 'nw')
     if p.multi, error('field "%s": "mv" is invalid for this category', name); end
end

% Convert to internal categories.
if cat(2) == 'a'
    if p.cache
        cat(2) = 't';
    end
elseif ismember(cat, {'cc'})
    if p.cache
        if p.private, error('field "%s": "private" and "cache" are mutually exclusive for this category', name); end
        if p.multi, error('field "%s": "cache" and "mv" are mutually exclusive for this category', name); end
    else
        p.private = (p.private || p.multi);
        cat = 'nc';
    end
elseif ismember(cat, {'cv'})
    if p.cache
        if p.private, error('field "%s": "private" and "cache" are mutually exclusive for this category', name); end
        if p.multi, error('field "%s": "cache" and "mv" are mutually exclusive for this category', name); end
    else
        p.private = (p.private || p.multi);
        if p.private
            cat = 'nv';
        else
            cat = 'nw';
        end
    end
end

if p.multi
    f = 'mvSyms';
else
    f = 'svSyms';
end

list     = d.cat.(cat).list;
mvList   = d.cat.(cat).mvList;
alwaysMV = ismember(cat(2), {'a', 't'});

d.cat.(cat).syms{end + 1} = name;
d.cat.(cat).(f){end + 1} = name;

if ~isempty(list)
    d.list.(list).syms{end + 1} = name;
    d.list.(list).(f){end + 1} = name;
end

if p.multi || alwaysMV
    d.list.(mvList).syms{end + 1} = name;
end

if p.multi || alwaysMV
    pos = numel(d.list.(mvList).syms);
elseif ~isempty(list)
    pos = numel(d.list.(list).svSyms);
else
    pos = numel(d.cat.(cat).svSyms);
end

s = struct;

s.cat     = cat;
s.field   = true;
s.group   = d.cat.(cat).group;
s.typeNo  = p.typeNo;
s.multi   = p.multi;
s.pos     = pos;
s.int     = p.int;
s.value   = NaN;
s.private = p.private;

if ismember(cat(2), {'a', 't'})
    try
        s = cns_trans('parse', s, p, false);
    catch
        error('field "%s": %s', name, cns_error);
    end
else
    if any(isfield(p, {'dims', 'dparts', 'dnames'}))
        error('field "%s": this category cannot specify dims/dparts/dnames', name);
    end
end

if (cat(1) == 's') && (d.synTypeNo == 0)
    error('field "%s": synapses are not defined for this type', name);
end

if cat(2) == 'z'
    if ~isfield(p, 'ptrType'), error('field "%s": missing type', name); end
    [ans, ptrTypeNo] = ismember(p.ptrType, h.def.types);
    if ptrTypeNo == 0, error('field "%s": invalid type', name); end
    if h.def.type.(p.ptrType).dimNo == 0
        error('field "%s": type "%s" does not define dims', name, p.ptrType);
    end
    if p.int == -1, error('field "%s": pointer fields must have numeric type "ind"', name); end
    d.refTypeNos = union(d.refTypeNos, ptrTypeNo);
    s.ptrTypeNo = ptrTypeNo;
    s.int       = -2;
else
    if isfield(p, 'ptrType')
        error('field "%s": this category cannot specify a type', name);
    end
end

res = h.(cat).res;
if ~isempty(res)
    [h, s.resNo] = GetResNo(h, res, sprintf('%s_%s', p.typeName, name));
end

d.syms{end + 1} = name;
d.sym.(name)    = s;

return;

%***********************************************************************************************************************

function [d, h, p] = SetDflt(d, h, name, p)

if ~isfield(d.sym, name)
    error('constant "%s_dflt": field "%s" is not defined', name, name);
end

cat = d.sym.(name).cat;

if ~d.sym.(name).field || ~h.(cat).dfltOK
    error('constant "%s_dflt": cannot define a default for symbol "%s"', name, name);
end

p.int = d.sym.(name).int;

p = CheckValue([name '_dflt'], p);

if (ndims(p.value) ~= 2) || all(size(p.value) > 1)
    error('constant "%s_dflt": invalid default value', name);
end

if d.sym.(name).multi
    if (h.(cat).dfltOK < 2) && (numel(p.value) > 1)
        error('constant "%s_dflt": invalid default value for field "%s"', name, name);
    end
    if isempty(p.value)
        p.value = [];
    else
        p.value = p.value(:)';
    end
else
    if numel(p.value) ~= 1
        error('constant "%s_dflt": invalid default value for field "%s"', name, name);
    end
end

d.sym.(name).value = p.value;

return;

%***********************************************************************************************************************

function [d, h] = InitConsts(d, h)

c = struct;

c.field = false;
c.syms  = {};

d.cats{end + 1} = 'c';
d.cat.c         = c;

return;

%***********************************************************************************************************************

function p = GetConstSymbol(name, rest)

if ~isempty(rest) && isnumeric(rest{1})
    p.value = rest{1};
    rest = rest(2 : end);
else
    error('constant "%s": invalid definition', name);
end

if ~isempty(rest) && ischar(rest{1}) && strcmp(rest{1}, 'int')
    p.int = -1;
    rest = rest(2 : end);
elseif ~isempty(rest) && ischar(rest{1}) && strcmp(rest{1}, 'ind')
    p.int = -2;
    rest = rest(2 : end);
else
    p.int = 0;
end

if ~isempty(rest)
    error('constant "%s": invalid definition', name);
end

return;

%***********************************************************************************************************************

function [d, h] = AddConst(d, h, name, p)

CheckName(d, h, name);

p = CheckValue(name, p);

d.cat.c.syms{end + 1} = name;

s = struct;

s.cat   = 'c';
s.field = false;
s.int   = p.int;
s.value = p.value;

d.syms{end + 1} = name;
d.sym.(name)    = s;

return;

%***********************************************************************************************************************

function [d, h] = InitArrays(d, h)

c = struct;

c.field = false;
c.syms  = {};

d.cats{end + 1} = 'a';
d.cat.a         = c;

return;

%***********************************************************************************************************************

function p = GetArraySymbol(name, rest)

if ~isempty(rest) && isnumeric(rest{1})
    p.size = rest{1};
    rest = rest(2 : end);
else
    error('array "%s": invalid definition', name);
end

if ~isempty(rest) && ischar(rest{1}) && strcmp(rest{1}, 'int')
    p.int = -1;
    rest = rest(2 : end);
elseif ~isempty(rest) && ischar(rest{1}) && strcmp(rest{1}, 'double')
    p.int = 3;
    rest = rest(2 : end);
else
    p.int = 0;
end

if ~isempty(rest)
    error('array "%s": invalid definition', name);
end

return;

%***********************************************************************************************************************

function [d, h] = AddArray(d, h, name, p)

CheckName(d, h, name);

d.cat.a.syms{end + 1} = name;

s = struct;

s.cat   = 'a';
s.field = false;
s.int   = p.int;
s.size  = p.size;

d.syms{end + 1} = name;
d.sym.(name)    = s;

return;

%***********************************************************************************************************************

function [d, h] = InitExtras(d, h)

c = struct;

c.field = false;
c.syms  = {};

d.cats{end + 1} = 'x';
d.cat.x         = c;

return;

%***********************************************************************************************************************

function p = GetExtraSymbol(name, rest)

if ~isempty(rest)
    error('symbol "%s": invalid definition', name);
end

p = struct;

return;

%***********************************************************************************************************************

function [d, h] = AddExtra(d, h, name, p)

CheckName(d, h, name);

d.cat.x.syms{end + 1} = name;

s = struct;

s.cat   = 'x';
s.field = false;

d.syms{end + 1} = name;
d.sym.(name)    = s;

return;

%***********************************************************************************************************************

function CheckName(d, h, name)

if any(strcmpi(name, h.badNames))
    error('symbol name "%s" is invalid', name);
end

if any(strcmpi(name, h.globalNames)) || any(strcmpi(name, d.syms))
    error('symbol name "%s" is already in use', name);
end

return;

%***********************************************************************************************************************

function p = CheckValue(name, p)

if isa(p.value, 'int32')
    if p.int == 0
        error('constant "%s": invalid numeric type; add an "int" modifier to define an integer', name);
    end
elseif ~isfloat(p.value)
    error('constant "%s": invalid numeric type', name);
end

p.value = double(p.value);

if ~all(isfinite(p.value(:)))
    error('constant "%s": invalid value', name);
end

if p.int < 0
    if any(p.value(:) ~= round(p.value(:))) || any(p.value(:) < cns_intmin) || any(p.value(:) > cns_intmax)
        error('constant "%s": invalid integer value', name);
    end
end

return;

%***********************************************************************************************************************

function [h, resNo] = GetResNo(h, res, key)

[ans, resNo] = ismember(key, h.([res 'List']));

if resNo == 0
    h.([res 'List']){end + 1} = key;
    resNo = numel(h.([res 'List']));
end

return;

%***********************************************************************************************************************

function d = Done(d, h)

d.dimCount  = numel(h.dimList);
d.maxSiz2p  = h.maxSiz2p;
d.shiftName = h.shiftName;

d.aList   = h.aList;
d.aCount  = numel(h.aList);
d.tList   = h.tList;
d.tCount  = numel(h.tList);
d.ccList  = h.ccList;
d.ccCount = numel(h.ccList);
d.cvList  = h.cvList;
d.cvCount = numel(h.cvList);
d.ncList  = h.ncList;
d.ncCount = numel(h.ncList);
d.nwList  = h.nwList;
d.nwCount = numel(h.nwList);

return;