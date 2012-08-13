function ki = cns_preprocess(package, tpath, abstract, ker, fout)

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

[b, ki] = ReadTop(package, tpath, abstract, ker, fout >= 0);

if fout < 0, return; end

if ki.pmode == 'n', return; end

path = '';
no   = 0;

for i = 1 : numel(b.lines)

    if ~strcmp(b.paths{i}, path)
        fprintf(fout, '#line %u "%s"\n', b.nos(i), strrep(b.paths{i}, '\', '\\'));
        path = b.paths{i};
        no   = b.nos  (i);
    elseif b.nos(i) ~= no
        fprintf(fout, '#line %u\n', b.nos(i));
        no = b.nos(i);
    end
    
    fprintf(fout, '%s\n', b.lines{i});

    no = no + 1;

end

return;

%***********************************************************************************************************************

function [b, ki] = ReadTop(package, tpath, abstract, ker, full)

b = Empty;

ki.pmode     = 'x';
ki.id        = '';
ki.blockSize = [];
ki.arrays    = {};
ki.array     = struct;
ki.iter      = {};

p.names = {};
p.parts = {};
p.used  = false(0);

for i = numel(tpath) : -1 : 1
    if strcmp(ker, 'dflt')
        id = tpath{i};
    else
        id = [tpath{i} '_' ker];
    end
    path = fullfile(fileparts(which([package '_' tpath{i}])), [package '_' id '.h']);
    if ~exist(path, 'file'), continue; end
    [ki, f] = ProcessFile(package, path, ~full, @ReadFileInfo, ki);
    if ki.pmode ~= 'p'
        ki.id = id;
        break;
    end
    if full
        p = ProcessFile('', f, true, @ReadParts, p);
    end
end

if (~abstract && (ki.pmode == 'x')) || (ki.pmode == 'p')
    error('type "%s", kernel "%s": no superclass defines a kernel', tpath{end}, ker);
end

if ~full, return; end

if ki.pmode == 'n'
    fclose(f.h);
else
    [b, p] = ProcessFile('', f, true, @ReadTemplate, ki.pmode, p);
end

n = find(~p.used, 1);
if ~isempty(n), error('type "%s", kernel "%s": #PART "%s" is not used', tpath{end}, ker, p.names{n}); end

return;

%***********************************************************************************************************************

function varargout = ProcessFile(package, path, close, func, varargin)

if ischar(path)

    h = fopen(path, 'r');
    if h < 0, error('unable to open "%s"', path); end

    f.package = package;
    f.path    = path;
    f.h       = h;
    f.line    = fgetl(h);
    f.no      = 1;

else

    f = path;

end

try
    [varargout{1 : nargout}] = func(f, varargin{:});
catch
    fclose(f.h);
    error('file "%s": %s', f.path, cns_error);
end

if close, fclose(f.h); end

return;

%***********************************************************************************************************************

function f = ReadLine(f)

f.line = fgetl(f.h);

f.no = f.no + 1;

return;

%***********************************************************************************************************************

function [ki, f] = ReadFileInfo(f, ki)

blockSizeFlag = false;

while true

    if isequal(f.line, -1), break; end
    [tok, rest] = strtok(f.line);

    if strcmp(tok, '#BLOCKSIZE')

        if blockSizeFlag, error('line %u: #BLOCKSIZE repeated', f.no); end
        blockSizeFlag = true;
        if isempty(ki.blockSize)
            [by, rest] = strtok(rest);
            [bx, rest] = strtok(rest);
            if ~isempty(strtrim(rest)), error('line %u: superfluous #BLOCKSIZE parameters', f.no); end
            ki.blockSize = [str2double(by), str2double(bx)];
            if ~all(isfinite(ki.blockSize) & (ki.blockSize >= 1) & (mod(ki.blockSize, 1) == 0))
                error('line %u: invalid #BLOCKSIZE', f.no);
            end
        end

    elseif strcmp(tok, '#ARRAY')

        [name, rest] = strtok(rest);
        [siz , rest] = strtok(rest);
        [dt  , rest] = strtok(rest);
        if ~isempty(strtrim(rest)), error('line %u: superfluous #ARRAY parameters', f.no); end
        if ~isvarname(name), error('line %u: invalid #ARRAY name', f.no); end
        if ismember(name, ki.arrays), error('line %u: #ARRAY name already in use', f.no); end
        siz = str2double(siz);
        if ~isfinite(siz) || (siz < 0) || (mod(siz, 1) ~= 0), error('line %u: invalid #ARRAY size', f.no); end
        dt = lower(dt);
        switch dt
        case ''      , dt = 0;
        case 'int'   , dt = -1;
        case 'double', dt = 3;
        otherwise    , error('line %u: invalid #ARRAY data type', f.no);
        end
        ki.arrays{end + 1} = name;
        ki.array.(name).size = siz;
        ki.array.(name).int  = dt;

    elseif strcmp(tok, '#ITERATE')

        [name, rest] = strtok(rest);
        if isempty(name), error('line %u: missing #ITERATE dimension', f.no); end
        if ismember(name, ki.iter), error('line %u: repeated #ITERATE dimension', f.no); end
        if ~isempty(strtrim(rest)), error('line %u: superfluous #ITERATE parameters', f.no); end
        ki.iter{end + 1} = name;

    elseif ~isempty(tok) && isempty(strmatch('//', tok))

        break;

    end

    f = ReadLine(f);

end

if isequal(f.line, -1)
    ki.pmode = 'k';
elseif strcmp(tok, '#NULL')
    ki.pmode = 'n';
elseif strcmp(tok, '#TEMPLATE')
    ki.pmode = 't';
elseif strcmp(tok, '#PART')
    ki.pmode = 'p';
else
    ki.pmode = 'k';
end

if (ki.pmode == 'p') && ~isempty(ki.iter)
    error('#ITERATE not allowed in #PART files');
end

return;

%***********************************************************************************************************************

function q = ReadParts(f, p)

q.names = {};
q.parts = {};
q.used  = false(0);

while true

    [ans, rest] = strtok(f.line);
    [name, rest] = strtok(rest);
    if isempty(name), error('line %u: #PART name missing', f.no); end
    if ~isempty(strtrim(rest)), error('line %u: #PART names cannot contain spaces', f.no); end

    if ismember(name, q.names), error('line %u: #PART name "%s" repeated', f.no, name); end

    [part, f, p] = Read(ReadLine(f), 'p', p, 0, 0);

    q.names{end + 1} = name;
    q.parts{end + 1} = part;
    q.used (end + 1) = false;

    if isequal(f.line, -1), break; end

end

keep = ~p.used & ~ismember(p.names, q.names);

q.names = cat(2, q.names, p.names(keep));
q.parts = cat(2, q.parts, p.parts(keep));
q.used  = cat(2, q.used , p.used (keep));

return;

%***********************************************************************************************************************

function [b, p] = ReadTemplate(f, pmode, p)

if pmode == 't'

    % The #TEMPLATE keyword is being kept for backwards compatibility.  Inside a #TEMPLATE, #PART means #INCLUDEPART.

    [ans, rest] = strtok(f.line);
    if ~isempty(strtrim(rest)), error('line %u: superfluous #TEMPLATE parameters', f.no); end

    f = ReadLine(f);

end

[b, f, p] = Read(f, pmode, p, 0, 0);

return;

%***********************************************************************************************************************

function [b, f, p] = Read(f, pmode, p, inc, level)

b = Empty;

while true

    if isequal(f.line, -1)
        if level > 0, error('line %u: missing #UNROLL_END', f.no); end
        break;
    end

    [tok, rest] = strtok(f.line);

    if strcmp(tok, '#UNROLL_START')

        [iter, rest] = strtok(rest);
        if isempty(iter), error('line %u: #UNROLL_START parameters missing', f.no); end

        [symbol, rest] = strtok(rest);
        lims = strtrim(rest);

        no1 = f.no;
        [c, f, p] = Read(ReadLine(f), pmode, p, inc, level + 1);

        b = Concat(b, Repeat(f.path, [no1 f.no], c, str2double(iter), symbol, lims, inc, level));

    elseif strcmp(tok, '#UNROLL_END')

        if ~isempty(strtrim(rest)), error('line %u: superfluous #UNROLL_END parameters', f.no); end

        if level == 0, error('line %u: unmatched #UNROLL_END', f.no); end

        break;

    elseif strcmp(tok, '#UNROLL_BREAK')

        if ~isempty(strtrim(rest)), error('line %u: superfluous #UNROLL_BREAK parameters', f.no); end

        if level == 0, error('line %u: unmatched #UNROLL_BREAK', f.no); end

        b = Add(b, f);

    elseif strcmp(tok, '#define')

        error('line %u: #define is not allowed', f.no);

    elseif strcmp(tok, '#include')

        error('line %u: #include is not allowed; use #INCLUDE', f.no);

    elseif strcmp(tok, '#INCLUDE')

        name = strtrim(rest);
        if isempty(name), error('line %u: #INCLUDE name missing', f.no); end

        path = which([f.package '_inc_' name '.h']);

        b = Concat(b, ProcessFile(f.package, path, true, @Read, pmode, p, inc + 1, 0));

    elseif strcmp(tok, '#TEMPLATE')

        error('line %u: #TEMPLATE is not allowed here', f.no);

    elseif strcmp(tok, '#PART') && any(pmode == 'kp')

        if pmode == 'k', error('line %u: #PART is not allowed here', f.no); end
        if inc > 0, error('line %u: #PART cannot appear inside an #INCLUDE file', f.no); end

        if level > 0, error('line %u: missing #UNROLL_END', f.no); end

        break;

    elseif ismember(tok, {'#PART', '#INCLUDEPART'})

        [name, rest] = strtok(rest);
        if isempty(name), error('line %u: #PART name missing', f.no); end
        if ~isempty(strtrim(rest)), error('line %u: #PART names cannot contain spaces', f.no); end

        n = find(strcmp(p.names, name), 1);
        if isempty(n), error('line %u: #PART "%s" is missing', f.no, name); end

        b = Concat(b, p.parts{n});

        p.used(n) = true;

    elseif ismember(tok, {'#NULL', '#BLOCKSIZE', '#ARRAY', '#ITERATE'})

        error('line %u: %s is not allowed here', f.no, tok);

    else

        b = Add(b, f);

    end

    f = ReadLine(f);

end

return;

%***********************************************************************************************************************

function b = Repeat(path, nos, c, iter, symbol, lims, inc, level)

if isempty(lims)
    b = RepeatSimple(path, nos, c, iter, symbol);
else
    b = RepeatComplex(path, nos, c, iter, symbol, lims, inc, level);
end

return;

%***********************************************************************************************************************

function b = RepeatSimple(path, nos, c, iter, symbol)

b = Empty;

b = Add(b, path, nos(1), 'for (int _ur = 0; _ur < 1; _ur++) {');

for i = 0 : iter - 1

    b = Add(b, path, nos(1), '{');

    for j = 1 : numel(c.lines)
        if strcmp(strtok(c.lines{j}), '#UNROLL_BREAK')
            b = Add(b, c.paths{j}, c.nos(j), 'break;');
        elseif isempty(symbol)
            b = Add(b, c, j);
        else
            b = Add(b, c.paths{j}, c.nos(j), strrep(c.lines{j}, symbol, sprintf('%u', i)));
        end
    end

    b = Add(b, path, nos(2), '}');

end

b = Add(b, path, nos(2), '}');

return;

%***********************************************************************************************************************

function b = RepeatComplex(path, nos, c, iter, symbol, lims, inc, level)

[args{1}, rest] = strtok(lims);
[args{2}, rest] = strtok(rest);
args{3} = strtrim(rest);
args = args(~strcmp(args, ''));
switch numel(args)
case 1, lim1 = '0'    ; cond = '<'    ; lim2 = args{1};
case 2, lim1 = args{1}; cond = '<'    ; lim2 = args{2};
case 3, lim1 = args{1}; cond = args{2}; lim2 = args{3};
end
switch cond
case '<' , extra = '0';
case '<=', extra = '1';
otherwise, error('invalid #UNROLL_START condition: "%s"', cond);
end

id = sprintf('%u_%u', inc, level);

b = Empty;

b = Add(b, path, nos(1), '{');
b = Add(b, path, nos(1), sprintf('int n_%s = ((%s) - (%s) + %s) / %u;', id, lim2, lim1, extra, iter));
b = Add(b, path, nos(1), sprintf('int i_%s = (%s);', id, lim1));
b = Add(b, path, nos(1), sprintf('for (int u_%s = 0; u_%s < n_%s; u_%s++) {', id, id, id, id));

for i = 1 : iter
    b = Add(b, path, nos(1), '{');
    for j = 1 : numel(c.lines)
        if strcmp(strtok(c.lines{j}), '#UNROLL_BREAK')
            b = Add(b, c.paths{j}, c.nos(j), sprintf('i_%s = (%s) + %s;', id, lim2, extra));
            b = Add(b, c.paths{j}, c.nos(j), 'break;');
        else
            b = Add(b, c.paths{j}, c.nos(j), strrep(c.lines{j}, symbol, sprintf('i_%s', id)));
        end
    end
    b = Add(b, path, nos(2), '}');
    b = Add(b, path, nos(2), sprintf('i_%s++;', id));
end

b = Add(b, path, nos(2), '}');
b = Add(b, path, nos(1), sprintf('for (; i_%s %s (%s); i_%s++) {', id, cond, lim2, id));

for j = 1 : numel(c.lines)
    if strcmp(strtok(c.lines{j}), '#UNROLL_BREAK')
        b = Add(b, c.paths{j}, c.nos(j), 'break;');
    else
        b = Add(b, c.paths{j}, c.nos(j), strrep(c.lines{j}, symbol, sprintf('i_%s', id)));
    end
end
    
b = Add(b, path, nos(2), '}');
b = Add(b, path, nos(2), '}');

return;

%***********************************************************************************************************************

function b = Empty

b.paths = {};
b.nos   = [];
b.lines = {};

return;

%***********************************************************************************************************************

function b = Add(b, varargin)

switch nargin
case 2
    f = varargin{1};
    path = f.path;
    no   = f.no;
    line = f.line;
case 3
    c = varargin{1};
    n = varargin{2};
    path = c.paths{n};
    no   = c.nos  (n);
    line = c.lines{n};
otherwise
    path = varargin{1};
    no   = varargin{2};
    line = varargin{3};
end

b.paths{end + 1} = path;
b.nos  (end + 1) = no;
b.lines{end + 1} = line;

return;

%***********************************************************************************************************************

function b = Concat(b1, b2)

b.paths = [b1.paths, b2.paths];
b.nos   = [b1.nos  , b2.nos  ];
b.lines = [b1.lines, b2.lines];

return;