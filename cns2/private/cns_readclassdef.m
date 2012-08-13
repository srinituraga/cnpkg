function [p, f] = cns_readclassdef(path, package, type)

if nargin < 3, type = ''; end

if isempty(type)
    cname = package;
else
    cname = [package '_' type];
end
if ~strcmp(fileparts(which(cname)), path), error('wrong "%s" in path', cname); end
fn = [cname '.m'];

c = meta.class.fromName(cname);
if ~isscalar(c), error('%s: unable to read classdef', fn); end

full = superclasses(cname);
direct = {};
for i = 1 : numel(c.SuperClasses)
    direct{i} = c.SuperClasses{i}.Name;
end

if any(ismember({package, [package '_']}, full)), error('%s: invalid supertype', fn); end

if isempty(type)
    if ~ismember('cns_package', direct), error('%s: must inherit directly from cns_package', fn); end
else
    if ismember('cns_package', full), error('%s: cannot inherit from cns_package', fn); end
end

if isempty(type)
    if ismember('cns_base', full), error('%s: cannot inherit from cns_base', fn); end
elseif strcmp(type, 'base')
    if ~ismember('cns_base', direct), error('%s: must inherit directly from cns_base', fn); end
else
    if ismember('cns_base', direct), error('%s: cannot inherit directly from cns_base', fn); end
end

if isempty(type) || strcmp(type, 'base')
    if ~isempty(strmatch([package '_'], full)), error('%s: cannot inherit from cell types', fn); end
    super = '';
else
    pos = strmatch([package '_'], direct);
    if isempty(pos), error('%s: must inherit directly from another cell type (perhaps the base type?)', fn); end
    if ~isscalar(pos), error('%s: cannot inherit from multiple cell types', fn); end
    super = direct{pos}(numel(package) + 2 : end);
    for i = [1 : pos - 1, pos + 1 : numel(direct)]
        % Make sure any other parent classes don't inherit from cell types.
        if ~isempty(strmatch([package '_'], superclasses(direct{i}))), error('%s: invalid parent type', fn); end
    end
end

props  = '';
fields = '';
init   = '';
for i = 1 : numel(c.Methods)
    m = c.Methods{i};
    if m.DefiningClass ~= c, continue; end
    if strcmpi(m.Name, 'cnsprops') || strcmpi(m.Name, 'cns_props')
        if isempty(type), error('%s: "%s" method not valid in this file', fn, m.Name); end
        if ~m.Static, error('%s: "%s" method must be static', fn, m.Name); end
        if ~isempty(props), error('%s: "%s" method duplicated', fn, m.Name); end
        props = m.Name;
    elseif strcmpi(m.Name, 'cnsfields') || strcmpi(m.Name, 'cns_fields')
        if ~m.Static, error('%s: "%s" method must be static', fn, m.Name); end
        if ~isempty(fields), error('%s: "%s" method duplicated', fn, m.Name); end
        fields = m.Name;
    elseif strcmpi(m.Name, 'cnsinit') || strcmpi(m.Name, 'cns_init')
        if ~isempty(type), error('%s: "%s" method not valid in this file', fn, m.Name); end
        if ~m.Static, error('%s: "%s" method must be static', fn, m.Name); end
        if ~isempty(init), error('%s: "%s" method duplicated', fn, m.Name); end
        if ~isempty(type), error('%s: "%s" method can only appear in %s.m', fn, m.Name, package); end
        init = m.Name;
    end
end

if isempty(props ), p = struct; else p = feval([cname '.' props ]); end
if isempty(fields), f = struct; else f = feval([cname '.' fields]); end

p.super = super;
p.init  = init;

return;