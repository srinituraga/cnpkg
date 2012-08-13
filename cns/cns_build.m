function cns_build(varargin)

% CNS_BUILD
%    Click <a href="matlab: cns_help('cns_build')">here</a> for help.

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

try
    Main(varargin{:});
catch
    error(cns_error);
end

end

%***********************************************************************************************************************

function Main(Package, varargin)

if nargin < 1, error('not enough arguments'); end

UserPath = fileparts(which([Package '_cns']));
if isempty(UserPath), error('cannot find "%s_cns.m"', Package); end
if exist(fullfile(UserPath, 'cns_build.m'), 'file'), error('invalid directory'); end

SourcePath = fullfile(fileparts(mfilename('fullpath')), 'source');

F = -1;
HelpMode = false;
Help = struct;
KeepGenerated = false;
Cleanup;

try
    Action(varargin{:});
catch
    Cleanup;
    rethrow(lasterror);
end

Cleanup;

%***********************************************************************************************************************

function Action(mode)

if nargin < 1, mode = 'compile'; end

def = cns_def(Package, true);

switch mode
case 'clean'

    CheckActive;

    DeleteMex('cuda');
    DeleteMex('cpu' );

    DeleteFile('%s_pre.cpp' , 'cuda');
    DeleteFile('%s_pre.cpp' , 'cpu' );
    DeleteFile('%s_info.txt', 'cuda');
    DeleteFile('help.txt');
    DeleteFile('def.mat' );

case 'compile'

    CheckActive;

    DeleteMex('cuda');
    DeleteMex('cpu' );

    Generate(def, 'cuda');
    Generate(def, 'cpu' );

    fprintf('compiling...\n');
    Compile('cuda', 'compile', '%s.cpp', ['%s.' mexext]);
    Compile('cpu' , 'compile', ['%s.' mexext]);

    cns_docpath(UserPath, fileparts(mfilename('fullpath')));

case 'generate'

    Generate(def, 'cuda');
    Generate(def, 'cpu' );

    KeepGenerated = true;

case 'preprocess'

    DeleteFile('%s_pre.cpp', 'cuda');
    DeleteFile('%s_pre.cpp', 'cpu' );

    Generate(def, 'cuda');
    Generate(def, 'cpu' );

    fprintf('compiling...\n');
    Compile('cuda', 'preprocess', '%s_pre.cpp');
    Compile('cpu' , 'preprocess', '%s_pre.cpp');

case 'info'

    DeleteFile('%s_info.txt', 'cuda');

    Generate(def, 'cuda');

    fprintf('compiling...\n');
    Compile('cuda', 'info', '%s_info.txt');

case 'help'

    DeleteFile('help.txt');

    GenHelp(def, 'cuda');

otherwise

    error('invalid action');

end

end

%***********************************************************************************************************************

function CheckActive

if numel(cns('sessions')) > 0
    error('please close any active cns sessions');
end

if mislocked('cns'), munlock('cns'); end
clear cns;
if mislocked('cns')
    error('the "cns" function is currently locked');
end

end

%***********************************************************************************************************************

function Generate(def, platform)

p.mainFilePath = UserFilePath(false, '%s.%s'       , platform, SrcExt(platform));
p.glbFilePath1 = UserFilePath(false, '%s_glb_dec.h', platform);
p.glbFilePath2 = UserFilePath(false, '%s_glb_def.h', platform);
p.sesFilePath  = UserFilePath(false, '%s_ses.h'    , platform);
p.decFilePath  = UserFilePath(false, '%s_dec.h'    , platform);
p.defFilePath  = UserFilePath(false, '%s_def.h'    , platform);
p.runFilePath  = UserFilePath(false, '%s_run.h'    , platform);

F = fopen(p.mainFilePath, 'w+');
if (F < 0), error('unable to write to "%s"', p.mainFilePath); end
GenerateMain(def, platform, p);
fclose(F);
F = -1;

F = fopen(p.glbFilePath1, 'w+');
if (F < 0), error('unable to write to "%s"', p.glbFilePath1); end
GenerateGlobal(def, platform, true);
fclose(F);
F = -1;

F = fopen(p.glbFilePath2, 'w+');
if (F < 0), error('unable to write to "%s"', p.glbFilePath2); end
GenerateGlobal(def, platform, false);
fclose(F);
F = -1;

F = fopen(p.sesFilePath, 'w+');
if (F < 0), error('unable to write to "%s"', p.sesFilePath); end
GenerateSession(def, platform);
fclose(F);
F = -1;

F = fopen(p.decFilePath, 'w+');
if (F < 0), error('unable to write to "%s"', p.decFilePath); end
GenerateKernelDec(def, platform);
fclose(F);
F = -1;

F = fopen(p.defFilePath, 'w+');
if (F < 0), error('unable to write to "%s"', p.defFilePath); end
GenerateKernelDef0;
for i = 1 : numel(def.types)
    GenerateKernelDef(def, platform, i);
end
fclose(F);
F = -1;

F = fopen(p.runFilePath, 'w+');
if (F < 0), error('unable to write to "%s"', p.runFilePath); end
GenerateKernelRun(def, platform);
fclose(F);
F = -1;

end

%***********************************************************************************************************************

function GenHelp(def, platform)

helpFilePath = UserFilePath(true, 'help.txt');

F = fopen(helpFilePath, 'w+');
if (F < 0), error('unable to write to "%s"', helpFilePath); end
HelpMode = true;

for i = 1 : numel(def.types)

    head = sprintf('**  TYPE: %s  **', def.types{i});
    line = repmat('*', 1, numel(head));

    fprintf(F, '%s\n', line);
    fprintf(F, '%s\n', head);
    fprintf(F, '%s\n', line);
    fprintf(F, '\n');

    Help.type = [];
    Help.num  = [];
    Help.sub  = [];
    Help.sym  = {};

    GenerateKernelDef(def, platform, i);

    [ans, inds] = sortrows([Help.type', Help.num', Help.sub']);
    Help.type = Help.type(inds);
    Help.num  = Help.num (inds);
    Help.sub  = Help.sub (inds);
    Help.sym  = Help.sym (inds);

    for j = 1 : numel(Help.sym)
        if (j > 1) && ((Help.type(j - 1) ~= Help.type(j)) || (Help.num(j - 1) ~= Help.num(j)))
            fprintf(F, '\n');
        end
        fprintf(F, '%s\n', Help.sym{j});
    end

    fprintf(F, '\n');

end

fclose(F);
F = -1;

end

%***********************************************************************************************************************

function GenerateMain(def, platform, p)

printf('#define _PACKAGE "%s"\n', Package);

if strcmp(platform, 'cuda')
    printf('#define _GPU\n');
end

printf('#define _USER_GLOBAL_DEC "%s"\n'    , p.glbFilePath1);
printf('#define _USER_GLOBAL_DEF "%s"\n'    , p.glbFilePath2);
printf('#define _USER_SESSION "%s"\n'       , p.sesFilePath );
printf('#define _USER_ALLKERNELS_DEC "%s"\n', p.decFilePath );
printf('#define _USER_ALLKERNELS_DEF "%s"\n', p.defFilePath );
printf('#define _USER_ALLKERNELS_RUN "%s"\n', p.runFilePath );

if isempty(which('cns_sync'))
    os = 'null';
else
    os = cns_osname;
end
printf('#define _SYNC_INC "sync_%s_inc.h"\n', os);
printf('#define _SYNC_DEC "sync_%s_dec.h"\n', os);
printf('#define _SYNC_DEF "sync_%s_def.h"\n', os);

printf('#include "%s"\n', fullfile(SourcePath, 'main.h'));

end

%***********************************************************************************************************************

function GenerateGlobal(def, platform, head)

if head

    [ans, bits] = cns_osname;
    printf('#define _BITS_%u\n', bits);

    printf('#define _VERSION %u\n', cns_version);

    e = cns_consts('layertable', def.maxSiz2p);
    names = fieldnames(e);
    for i = 1 : numel(names)
        printf('#define _LT_%s %u\n', upper(names{i}), e.(names{i}));
    end

    for i = 2 : numel(def.types)
        d = def.type.(def.types{i});
        WriteZClasses(def.types{i}, d.super);
    end

    printf('#define _G_CDATASYM _g_%s_cData\n'  , Package);
    printf('#define _G_CMETASYM _g_%s_cMeta\n'  , Package);
    printf('#define _G_CDATASTR "_g_%s_cData"\n', Package);
    printf('#define _G_CMETASTR "_g_%s_cMeta"\n', Package);

    printf('#define _NUM_T %u\n' , def.tCount );
    printf('#define _NUM_CC %u\n', def.ccCount);
    printf('#define _NUM_CV %u\n', def.cvCount);

    printf('#define _NUM_T_NZ %u\n' , max(def.tCount , 1));
    printf('#define _NUM_CC_NZ %u\n', max(def.ccCount, 1));

    % The following constant defines the size of several arrays that are members of a structure that is passed as a
    % kernel argument.  It used to have minimum value 1.  It now has minimum value 2 because of what I assume is a
    % compiler bug in CUDA 3.0.  The structure's contents were not being passed correctly.
    printf('#define _NUM_CV_NZ %u\n', max(def.cvCount, 2));

    WriteAllTexRefs(def, platform, true);

end

WriteArrayTrans(def, platform, head, 'ma');
WriteTexTrans  (def, platform, head, 'mt');
for i = 1 : numel(def.types)
    d = def.type.(def.types{i});
    WriteArrayTrans(d, platform, head, 'ga');
    WriteTexTrans  (d, platform, head, 'gt');
    WriteArrayTrans(d, platform, head, 'la');
    WriteTexTrans  (d, platform, head, 'lt');
end

sList  = [];
rsList = {};
for i = 1 : numel(def.types)
    d = def.type.(def.types{i});
    if d.dimNo == 0, continue; end
    if ~ismember(d.dimNo, sList)
        WriteLayerTrans(d, platform, head);
        sList(end + 1) = d.dimNo;
    end
    for j = 1 : numel(d.cat.cc.syms)
        name = d.cat.cc.syms{j};
        resNo = d.sym.(name).resNo;
        key = sprintf('c%u.%u', resNo, d.dimNo);
        if ~ismember(key, rsList)
            WriteCVTrans(d, platform, head, '_CTrans', sprintf('_g_%s_tCData', Package), resNo, false);
            rsList{end + 1} = key;
        end
    end
    for j = 1 : numel(d.cat.cv.syms)
        name = d.cat.cv.syms{j};
        resNo = d.sym.(name).resNo;
        key = sprintf('v%u.%u', resNo, d.dimNo);
        if ~ismember(key, rsList)
            WriteCVTrans(d, platform, head, '_VTrans', sprintf('_g_%s_tVData', Package), resNo, true);
            rsList{end + 1} = key;
        end
    end
    for j = 1 : numel(d.cat.nc.svSyms)
        name = d.cat.nc.svSyms{j};
        if ~d.sym.(name).private
            resNo = d.sym.(name).resNo;
            key = sprintf('n%u.%u', resNo, d.dimNo);
            if ~ismember(key, rsList)
                WriteNTrans(d, platform, head, '_NTrans', d.sym.(name).pos, resNo, 'n');
                rsList{end + 1} = key;
            end
        end
    end
    for j = 1 : numel(d.cat.nw.svSyms)
        name = d.cat.nw.svSyms{j};
        resNo = d.sym.(name).resNo;
        key = sprintf('w%u.%u', resNo, d.dimNo);
        if ~ismember(key, rsList)
            WriteNTrans(d, platform, head, '_WTrans', d.sym.(name).pos, resNo, 'w');
            rsList{end + 1} = key;
        end
    end
end

end

%***********************************************************************************************************************

function GenerateSession(def, platform)

WriteAllTexRefs(def, platform, false);

end

%***********************************************************************************************************************

function WriteZClasses(name, super)

printf('class _T_%s : public _T_%s {\n', upper(name), upper(super));
printf('};\n');

printf('INLINE _T_%s _P_%s(int z) {\n', upper(name), upper(name));
printf('_T_%s p;\n', upper(name));
printf('p.z = z;\n');
printf('return p;\n');
printf('}\n');

printf('INLINE int _Z_%s(_T_%s p) {\n', upper(name), upper(name));
printf('return p.z;\n');
printf('}\n');

end

%***********************************************************************************************************************

function WriteAllTexRefs(d, platform, glb)

WriteTexRefs(platform, glb, '_TEXTUREA', sprintf('_g_%s_tTDataA', Package), d.tCount , 'GetTTexA');
WriteTexRefs(platform, glb, '_TEXTUREB', sprintf('_g_%s_tTDataB', Package), d.tCount , 'GetTTexB');
WriteTexRefs(platform, glb, '_TEXTUREA', sprintf('_g_%s_tCDataA', Package), d.ccCount, 'GetCTexA');
WriteTexRefs(platform, glb, '_TEXTUREB', sprintf('_g_%s_tCDataB', Package), d.ccCount, 'GetCTexB');
WriteTexRefs(platform, glb, '_TEXTUREA', sprintf('_g_%s_tVDataA', Package), d.cvCount, 'GetVTexA');
WriteTexRefs(platform, glb, '_TEXTUREB', sprintf('_g_%s_tVDataB', Package), d.cvCount, 'GetVTexB');

end

%***********************************************************************************************************************

function WriteTexRefs(platform, glb, type, name, count, func)

if (strcmp(platform, 'cuda') && strcmp(type, '_TEXTUREB')) == glb

    if ~glb, printf('public:\n'); end

    for i = 1 : count
        printf('%s %s%u;\n', type, name, i - 1);
    end

end

if ~glb

    printf('public:\n');

    printf('%s *%s(unsigned int f) {\n', type, func);

    if count == 0

        printf('return NULL;\n');

    else

        printf('switch (f) {\n');
        for i = 1 : count
            printf('case %u: return &%s%u;\n', i - 1, name, i - 1);
        end
        printf('default: return NULL;\n');
        printf('}\n');

    end

    printf('}\n');
    
end

end

%***********************************************************************************************************************

function WriteArrayTrans(d, platform, head, cat)

for i = 1 : numel(d.cat.(cat).syms)

    name = d.cat.(cat).syms{i};
    if d.sym.(name).typeNo ~= d.typeNo, continue; end

    cname = sprintf('_ALEN_%u' , d.sym.(name).resNo - 1);
    fname = sprintf('_ATrans%u', d.sym.(name).resNo - 1);

    entryLen = ceil((2 + sum(cellfun(@numel, d.sym.(name).dims)) + 2) / 2) * 2;

    if head
        printf('#define %s %u\n', cname, entryLen);
    end

    cns_transgen('handle', d.sym.(name), @printf, platform, fname, 'a', head);

    if ~head
        cns_transgen('size'  , d.sym.(name), @printf, platform, fname);
        cns_transgen('lookup', d.sym.(name), @printf, platform, fname, 'a');
    end

end

end

%***********************************************************************************************************************

function WriteTexTrans(d, platform, head, cat)

for i = 1 : numel(d.cat.(cat).syms)

    name = d.cat.(cat).syms{i};
    if d.sym.(name).typeNo ~= d.typeNo, continue; end

    cname = sprintf('_TLEN_%u' , d.sym.(name).resNo - 1);
    fname = sprintf('_TTrans%u', d.sym.(name).resNo - 1);

    tnb = sprintf('_g_%s_tTDataB%u', Package, d.sym.(name).resNo - 1);

    entryLen = ceil((2 + sum(cellfun(@numel, d.sym.(name).dims)) + 2) / 2) * 2;

    if head
        printf('#define %s %u\n', cname, entryLen);
    end

    cns_transgen('handle', d.sym.(name), @printf, platform, fname, 't', head);

    if ~head
        cns_transgen('size'  , d.sym.(name), @printf, platform, fname);
        cns_transgen('lookup', d.sym.(name), @printf, platform, fname, 't', tnb);
    end

end

end

%***********************************************************************************************************************

function WriteLayerTrans(d, platform, head)

fname = sprintf('_LTrans%u', d.dimNo - 1);

if ~head
    cns_transgen('size' , d, @printf, platform, fname, false);
    cns_transgen('coord', d, @printf, platform, fname);
end

end

%***********************************************************************************************************************

function WriteCVTrans(d, platform, head, fname, texname, resNo, write)

fn  = sprintf('%s%us%u', fname  , resNo - 1, d.dimNo - 1);
tnb = sprintf('%sB%u'  , texname, resNo - 1);

cns_transgen('handle', d, @printf, platform, fn, 'c', head);

if ~head
    cns_transgen('size'  , d, @printf, platform, fn, true);
    cns_transgen('lookup', d, @printf, platform, fn, 'c', tnb);
    if write
        cns_transgen('write', d, @printf, platform, fn);
    end
end

end

%***********************************************************************************************************************

function WriteNTrans(d, platform, head, fname, pos, resNo, type)

fn = sprintf('%s%us%u', fname, resNo - 1, d.dimNo - 1);

cns_transgen('handle', d, @printf, platform, fn, type, head);

if ~head
    cns_transgen('size'  , d, @printf, platform, fn, true);
    cns_transgen('lookup', d, @printf, platform, fn, type, pos);
end

end

%***********************************************************************************************************************

function GenerateKernelDec(def, platform)

for i = 1 : numel(def.types)

    name = def.types{i};
    if ~def.type.(name).kernel, continue; end

    printf('#define _USER_KERNEL_NAME _kernel_%s\n', name);
    printf('#include "%s"\n', fullfile(SourcePath, 'kernel_dec.h'));
    printf('#undef _USER_KERNEL_NAME\n');

end

end

%***********************************************************************************************************************

function GenerateKernelDef0

filePath = fullfile(UserPath, sprintf('%s_cns.h', Package));

if exist(filePath, 'file')
    printf('#include "%s"\n', filePath);
end

end

%***********************************************************************************************************************

function GenerateKernelDef(def, platform, i)

name = def.types{i};
d = def.type.(name);
if ~HelpMode && ~d.kernel, return; end

a = ArrayInfo(d, platform);

WriteCommonFields(true, def);
WriteConsts      (true, def);
WriteKernelFields(true, def, i);
WriteConsts      (true, d);

if d.synTypeNo ~= 0
    printf('#define _USES_SYNS\n');
end

if a.int32Size > 0
    printf('#define _USES_INT32_ARRAYS\n');
    printf('#define _INT32_ARRAY_SIZE %u\n', a.int32Size);
end
if a.floatSize > 0
    printf('#define _USES_FLOAT_ARRAYS\n');
    printf('#define _FLOAT_ARRAY_SIZE %u\n', a.floatSize);
end
if a.doubleSize > 0
    printf('#define _USES_DOUBLE_ARRAYS\n');
    printf('#define _DOUBLE_ARRAY_SIZE %u\n', a.doubleSize);
end
for j = 1 : numel(a.names)
    printf(0, SymNum(d, a.names{j}), '#define %s(E) %s(%u,E)\n', upper(a.names{j}), a.macros{j}, a.offsets(j));
end

kerFilePath1 = fullfile(UserPath, [Package '_cns']);
kerFilePath2 = UserFilePath(false, '%s_def_%s.h', platform, name);
if ~HelpMode, cns_preprocess(kerFilePath1, d.typePath, kerFilePath2); end

printf('#define _USER_KERNEL_NAME _kernel_%s\n', name);
printf('#define _USER_KERNEL_NAME2 _kernel2_%s\n', name);
printf('#define _USER_KERNEL_DEF "%s"\n', kerFilePath2);
printf('#include "%s"\n', fullfile(SourcePath, 'kernel_def.h'));
printf('#undef _USER_KERNEL_NAME\n');
printf('#undef _USER_KERNEL_NAME2\n');
printf('#undef _USER_KERNEL_DEF\n');

WriteCommonFields(false, def);
WriteConsts      (false, def);
WriteKernelFields(false, def, i);
WriteConsts      (false, d);

if d.synTypeNo ~= 0
    printf('#undef _USES_SYNS\n');
end

if a.int32Size > 0
    printf('#undef _USES_INT32_ARRAYS\n');
    printf('#undef _INT32_ARRAY_SIZE\n');
end
if a.floatSize > 0
    printf('#undef _USES_FLOAT_ARRAYS\n');
    printf('#undef _FLOAT_ARRAY_SIZE\n');
end
if a.doubleSize > 0
    printf('#undef _USES_DOUBLE_ARRAYS\n');
    printf('#undef _DOUBLE_ARRAY_SIZE\n');
end
for j = 1 : numel(a.names)
    printf('#undef %s\n', upper(a.names{j}));
end

end

%***********************************************************************************************************************

function WriteCommonFields(on, def)

q.on     = on;
q.names  = def.types;
q.dc     = struct;
q.typeNo = [];
q.ref    = '';
q.refNo  = -1;
q.d      = def;

WriteCFields (q, 'mp', 'M');
WritePFields (q, 'mz', 'M');
WriteATFields(q, 'ma', 'M', 'Array');
WriteATFields(q, 'mt', 'M', 'Tex');

WriteDef(q, 'ITER_NO' , '_ITER_NO');
WriteDef(q, 'PHASE_NO', '_PHASE_NO');

WriteDef(q, 'ERROR(...)', '_ERROR(__VA_ARGS__)');
WriteDef(q, 'PRINT(...)', '_PRINT(__VA_ARGS__)');

end

%***********************************************************************************************************************

function WriteKernelFields(on, def, i)

q.on     = on;
q.names  = def.types;
q.dc     = struct;
q.typeNo = i;
q.ref    = '';
q.refNo  = 0;
q.d      = def.type.(def.types{i});

WriteField(q, '', struct, 'THIS_Z', false, '_P_%U(_THIS_Z)');

WriteCoords(q, '', struct, 'THIS_%C', false, '_GetCoord(%S,%D)'  );
WriteCoords(q, '', struct, '%C_SIZE', false, '_GetLayerSz(%S,%D)');

for j = 1 : numel(q.d.dims)
    if ~q.d.dmap(j), continue; end
    dname = q.d.dnames{j};
    dno = sum(cellfun(@numel, q.d.dims(1 : j - 1)));
    pos = q.d.sym.([dname '_start']).pos - 1;
    WriteField(q, '', struct, 'THIS_%1_CENTER', false, '_GetCenter(%S,%2,%3)', upper(dname), dno, pos);
    if q.d.dmap(j) == 2
        pos = q.d.sym.([dname '_shift']).pos - 1;
        WriteField(q, '', struct, 'THIS_%1_NEW', false, '_GetCoordNew(%S,%2,%3)', upper(dname), dno, pos);
    end
end

WriteCFields (q, 'gp', 'G');
WritePFields (q, 'gz', 'G');
WriteATFields(q, 'ga', 'G', 'Array');
WriteATFields(q, 'gt', 'G', 'Tex');
WriteCFields (q, 'lp', 'L');
WritePFields (q, 'lz', 'L');
WriteATFields(q, 'la', 'L', 'Array');
WriteATFields(q, 'lt', 'L', 'Tex');

WriteCVFields(q, 'cc', false, 'Const');
WriteCVFields(q, 'cv', true , 'Var');

WriteNFields(q, 'nc', false, 'N');
WriteNFields(q, 'nv', true , 'N');
WriteNFields(q, 'nw', true , 'W');

if q.d.synTypeNo ~= 0

    WriteDef(q, 'NUM_SYN'      , '_NUM_SYN');
    WriteDef(q, 'SELECT_SYN(E)', '_SELECT_SYN(E)');
    WriteDef(q, 'SYN_Z'        , sprintf('_P_%s(_SYN_Z)', upper(q.names{q.d.synTypeNo})));
    WriteDef(q, 'SYN_TYPE'     , '_SYN_TYPE');

    WriteFields(q, 'sc', 's', 'READ_%N'      , true , '_GetSField(%F)');
    WriteFields(q, 'sc', 'm', 'READ_%N(E)'   , true , '_GetSFieldMV(%M,%N,E)');
    WriteFields(q, 'sc', 'm', 'NUM_%N'       , false, '_LMVCount(%M)');

    WriteFields(q, 'sv', 's', 'READ_%N'      , true , '_GetSField(%F)');
    WriteFields(q, 'sv', 'm', 'READ_%N(E)'   , true , '_GetSFieldMV(%M,%N,E)');
    WriteFields(q, 'sv', 's', 'WRITE_%N(V)'  , false, '_SetSField(%F,%V)');
    WriteFields(q, 'sv', 'm', 'WRITE_%N(E,V)', false, '_SetSFieldMV(%M,%N,E,%V)');
    WriteFields(q, 'sv', 'm', 'NUM_%N'       , false, '_LMVCount(%M)');

end

q.dc = def.type.(def.types{i});

allRefs = false(1, numel(def.types));
for j = q.dc.refTypeNos
    allRefs = allRefs | def.type.(def.types{j}).isType;
end
allRefs = find(allRefs);

for j = allRefs

    q.typeNo = j;
    q.ref    = 't';
    q.refNo  = j;
    q.d      = def.type.(def.types{j});

    WriteRelFields(q);

end

end

%***********************************************************************************************************************

function WriteRelFields(q)

if q.typeNo == q.dc.synTypeNo

    WriteCoords(q, '', struct, 'SYN_%C', false, '_GetPCoord(%S,%D)');

end

if ismember(q.typeNo, q.dc.refTypeNos)

    WriteField(q, '', struct, '%T_PTR', false, '_T_%U');

end

if q.d.dimTypeNo == q.typeNo

    WriteCoords(q, '', struct, '%T_%C_SIZE(Z)', false, '_GetZLayerSz(%Z,%S,%D)');

    for j = 1 : numel(q.d.dims)
        if ~q.d.dmap(j), continue; end
        dname = q.d.dnames{j};
        pos   = q.d.sym.([dname '_start']).pos - 1;
        WriteField(q, '', struct, '%T_%1_CENTER(Z,C)', false, '_GetZCenter(%Z,%2,C)', upper(dname), pos);
    end

    for j = 1 : numel(q.dc.dims)

        if ~q.dc.dmap(j), continue; end

        dn = q.dc.dnames{j};
        [ans, k] = ismember(dn, q.d.dnames);
        if k == 0, continue; end
        if ~q.d.dmap(k), continue; end

        s1 = q.dc.dimNo - 1;
        s2 = q.d .dimNo - 1;
        d1 = sum(cellfun(@numel, q.dc.dims(1 : j - 1)));
        d2 = sum(cellfun(@numel, q.d .dims(1 : k - 1)));
        p1 = q.dc.sym.([dn '_start']).pos - 1;
        p2 = q.d .sym.([dn '_start']).pos - 1;

        a  = {upper(dn), s1, d1, p1, s2, d2, p2};
        aa = {upper(dn), s2, d2, p2};

        WriteField(q, '', struct, 'GET_%T_%1_RF_DIST(Z,R,...)'     , false, '_GetRFDist(%Z,%2,%3,%4,%5,%6,%7,R,%A)', a {:});
        WriteField(q, '', struct, 'GET_%T_%1_RF_NEAR(Z,N,...)'     , false, '_GetRFNear(%Z,%2,%3,%4,%5,%6,%7,N,%A)', a {:});
        WriteField(q, '', struct, 'GET_%T_%1_RF_DIST_AT(Z,P,R,...)', false, '_GetRFDistAt(%Z,%2,%3,%4,P,R,%A)'     , aa{:});
        WriteField(q, '', struct, 'GET_%T_%1_RF_NEAR_AT(Z,P,N,...)', false, '_GetRFNearAt(%Z,%2,%3,%4,P,N,%A)'     , aa{:});

    end

end

WriteRelCFields (q, 'gp', 'G');
WriteRelATFields(q, 'ga', 'G', 'Array');
WriteRelATFields(q, 'gt', 'G', 'Tex');
WriteRelCFields (q, 'lp', 'L');
WriteRelATFields(q, 'la', 'L', 'Array');
WriteRelATFields(q, 'lt', 'L', 'Tex');

WriteRelCVFields(q, 'cc', 'Const');
WriteRelCVFields(q, 'cv', 'Var');

WriteRelNFields(q, 'nc', 'N');
WriteRelNFields(q, 'nv', 'N');
WriteRelNFields(q, 'nw', 'W');

end

%***********************************************************************************************************************

function WriteCFields(q, cat, varargin)

WriteFields(q, cat, 's', '%N'    , true , '_Get%1Const(%F)'       , varargin{:});
WriteFields(q, cat, 'm', '%N(E)' , true , '_Get%1ConstMV(%M,%N,E)', varargin{:});
WriteFields(q, cat, 'm', 'NUM_%N', false, '_%1MVCount(%M)'        , varargin{:});

end

%***********************************************************************************************************************

function WriteRelCFields(q, cat, varargin)

WriteFields(q, cat, 'sp', '%T_%N(Z)'    , true , '_GetZ%1Const(%Z,%F)'       , varargin{:});
WriteFields(q, cat, 'mp', '%T_%N(Z,E)'  , true , '_GetZ%1ConstMV(%Z,%M,%N,E)', varargin{:});
WriteFields(q, cat, 'mp', 'NUM_%T_%N(Z)', false, '_Z%1MVCount(%Z,%M)'        , varargin{:});

end

%***********************************************************************************************************************

function WritePFields(q, cat, varargin)

WriteFields(q, cat, 's', '%N'    , false, '_P_%P(__float_as_int(_Get%1Const(%F)))'       , varargin{:});
WriteFields(q, cat, 'm', '%N(E)' , false, '_P_%P(__float_as_int(_Get%1ConstMV(%M,%N,E)))', varargin{:});
WriteFields(q, cat, 'm', 'NUM_%N', false, '_%1MVCount(%M)'                               , varargin{:});

end

%***********************************************************************************************************************

function WriteATFields(q, cat, varargin)

va = varargin;

WriteFields(q, cat, 's', '%N_%C_SIZE'                  , false, '_Get%1%2Sz(%R,%M,%N,0,%D)', va{:});
WriteFields(q, cat, 'm', '%N_%C_SIZE(E)'               , false, '_Get%1%2Sz(%R,%M,%N,E,%D)', va{:});
WriteFields(q, cat, 's', 'READ_%N(%X)'                 , true , '_Get%1%2(%R,%M,%N,0,%X)'  , va{:});
WriteFields(q, cat, 'm', 'READ_%N(E,%X)'               , true , '_Get%1%2(%R,%M,%N,E,%X)'  , va{:});
WriteFields(q, cat, 'm', 'NUM_%N'                      , false, '_%1MVCount(%M)'           , va{:});
WriteFields(q, cat, 's', 'GET_%N_HANDLE'               , false, '_Get%1%2H(%R,%M,%N,0)'    , va{:});
WriteFields(q, cat, 'm', 'GET_%N_HANDLE(E)'            , false, '_Get%1%2H(%R,%M,%N,E)'    , va{:});
WriteFields(q, cat, '' , '%N_HANDLE'                   , false, '_Def%2H(%R)'              , va{:});
WriteFields(q, cat, '' , '%N_HANDLE_%C_SIZE(H)'        , false, '_GetH%2Sz(%R,H,%D)'       , va{:});
WriteFields(q, cat, '' , 'READ_%N_HANDLE(H,%X)'        , true , '_GetH%2(%R,H,%X)'         , va{:});
WriteFields(q, cat, '' , 'GET_%N_HANDLE_IPOS(H,%X,Y,X)', false, '_GetH%21(%R,H,%X,Y,X)'    , va{:});

if strcmp(varargin{2}, 'Array')

    WriteFields(q, cat, '' , 'READ_%N_IPOS(H,Y,X)'  , true , '_GetArray2(%R,H,Y,X)'        , va{:});

else

    WriteFields(q, cat, 's', 'GET_%N_IPOS(%X,Y,X)'  , false, '_Get%1%21(%R,%M,%N,0,%X,Y,X)', va{:});
    WriteFields(q, cat, 'm', 'GET_%N_IPOS(E,%X,Y,X)', false, '_Get%1%21(%R,%M,%N,E,%X,Y,X)', va{:});
    WriteFields(q, cat, '' , 'READ_%N_IPOS(Y,X)'    , true , '_GetTex2(%R,Y,X)'            , va{:});

end

end

%***********************************************************************************************************************

function WriteRelATFields(q, cat, varargin)

va = varargin;

WriteFields(q, cat, 'sp', '%T_%N_%C_SIZE(Z)'            , false, '_GetZ%1%2Sz(%Z,%R,%M,%N,0,%D)', va{:});
WriteFields(q, cat, 'mp', '%T_%N_%C_SIZE(Z,E)'          , false, '_GetZ%1%2Sz(%Z,%R,%M,%N,E,%D)', va{:});
WriteFields(q, cat, 'sp', 'READ_%T_%N(Z,%X)'            , true , '_GetZ%1%2(%Z,%R,%M,%N,0,%X)'  , va{:});
WriteFields(q, cat, 'mp', 'READ_%T_%N(Z,E,%X)'          , true , '_GetZ%1%2(%Z,%R,%M,%N,E,%X)'  , va{:});
WriteFields(q, cat, 'mp', 'NUM_%T_%N(Z)'                , false, '_Z%1MVCount(%Z,%M)'           , va{:});
WriteFields(q, cat, 'sp', 'GET_%T_%N_HANDLE(Z)'         , false, '_GetZ%1%2H(%Z,%R,%M,%N,0)'    , va{:});
WriteFields(q, cat, 'mp', 'GET_%T_%N_HANDLE(Z,E)'       , false, '_GetZ%1%2H(%Z,%R,%M,%N,E)'    , va{:});
WriteFields(q, cat, 'p' , '%H_HANDLE'                   , false, '_Def%2H(%R)'                  , va{:});
WriteFields(q, cat, 'p' , '%H_HANDLE_%C_SIZE(H)'        , false, '_GetH%2Sz(%R,H,%D)'           , va{:});
WriteFields(q, cat, 'p' , 'READ_%H_HANDLE(H,%X)'        , true , '_GetH%2(%R,H,%X)'             , va{:});
WriteFields(q, cat, 'p' , 'GET_%H_HANDLE_IPOS(H,%X,Y,X)', false, '_GetH%21(%R,H,%X,Y,X)'        , va{:});

if strcmp(varargin{2}, 'Array')

    WriteFields(q, cat, 'p' , 'READ_%H_IPOS(H,Y,X)'       , true , '_GetArray2(%R,H,Y,X)'            , va{:});

else

    WriteFields(q, cat, 'sp', 'GET_%T_%N_IPOS(Z,%X,Y,X)'  , false, '_GetZ%1%21(%Z,%R,%M,%N,0,%X,Y,X)', va{:});
    WriteFields(q, cat, 'mp', 'GET_%T_%N_IPOS(Z,E,%X,Y,X)', false, '_GetZ%1%21(%Z,%R,%M,%N,E,%X,Y,X)', va{:});
    WriteFields(q, cat, 'p' , 'READ_%H_IPOS(Y,X)'         , true , '_GetTex2(%R,Y,X)'                , va{:});

end

end

%***********************************************************************************************************************

function WriteCVFields(q, cat, write, varargin)

WriteFields(q, cat, '', 'READ_%N', true, '_GetC%1(%B,%F)', varargin{:});

if write

    WriteFields(q, cat, '', 'WRITE_%N(V)', false, '_SetC%1(%B,%R,%F,%V)', varargin{:});

end

end

%***********************************************************************************************************************

function WriteRelCVFields(q, cat, varargin)

va = varargin;

if q.typeNo == q.dc.synTypeNo

    WriteFields(q, cat, '', 'READ_PRE_%N', true, '_GetPC%1(%B,%F)', va{:});

end

q.ref = 'd';

WriteFields(q, cat, '', 'READ_%T_%N(Z,%X)'            , true , '_GetZC%1(%Z,%B,%F,%X)'     , va{:});
WriteFields(q, cat, '', 'GET_%T_%N_HANDLE(Z)'         , false, '_GetZC%1H(%Z,%B,%F)'       , va{:});
WriteFields(q, cat, '', 'GET_%T_%N_IPOS(Z,%X,Y,X)'    , false, '_GetZC%11(%Z,%B,%F,%X,Y,X)', va{:});
WriteFields(q, cat, '', '%I_HANDLE'                   , false, '_DefC%1H(%B)'              , va{:});
WriteFields(q, cat, '', '%I_HANDLE_%C_SIZE(H)'        , false, '_GetHC%1Sz(H,%B,%D)'       , va{:});
WriteFields(q, cat, '', 'READ_%I_HANDLE(H,%X)'        , true , '_GetHC%1(H,%B,%X)'         , va{:});
WriteFields(q, cat, '', 'GET_%I_HANDLE_IPOS(H,%X,Y,X)', false, '_GetHC%11(H,%B,%X,Y,X)'    , va{:});
WriteFields(q, cat, '', 'READ_%I_IPOS(Y,X)'           , true , '_GetC%12(%B,Y,X)'          , va{:});

end

%***********************************************************************************************************************

function WriteNFields(q, cat, write, varargin)

va = varargin;

WriteFields(q, cat, 's', 'READ_%N'   , true , '_Get%1Field(%F)'       , va{:});
WriteFields(q, cat, 'm', 'READ_%N(E)', true , '_Get%1FieldMV(%M,%N,E)', va{:});
WriteFields(q, cat, 'm', 'NUM_%N'    , false, '_LMVCount(%M)'         , va{:});

if write

    WriteFields(q, cat, 's', 'WRITE_%N(V)'  , false, '_Set%1Field(%F,%V)'       , va{:});
    WriteFields(q, cat, 'm', 'WRITE_%N(E,V)', false, '_Set%1FieldMV(%M,%N,E,%V)', va{:});

end

end

%***********************************************************************************************************************

function WriteRelNFields(q, cat, varargin)

va = varargin;

if q.typeNo == q.dc.synTypeNo

    WriteFields(q, cat, 'sp', 'READ_PRE_%N', true, '_GetP%1Field(%B)', va{:});

end

q.ref = 'd';

WriteFields(q, cat, 'sp', 'READ_%T_%N(Z,%X)'            , true , '_GetZ%1Field(%Z,%B,%X)'    , va{:});
WriteFields(q, cat, 'sp', 'GET_%T_%N_HANDLE(Z)'         , false, '_GetZ%1FieldH(%Z,%B)'      , va{:});
WriteFields(q, cat, 'sp', '%I_HANDLE'                   , false, '_Def%1FieldH(%B)'          , va{:});
WriteFields(q, cat, 'sp', '%I_HANDLE_%C_SIZE(H)'        , false, '_GetH%1FieldSz(H,%B,%D)'   , va{:});
WriteFields(q, cat, 'sp', 'READ_%I_HANDLE(H,%X)'        , true , '_GetH%1Field(H,%B,%X)'     , va{:});
WriteFields(q, cat, 'sp', 'GET_%I_HANDLE_IPOS(H,%X,Y,X)', false, '_GetH%1Field1(H,%B,%X,Y,X)', va{:});
WriteFields(q, cat, 'sp', 'READ_%I_IPOS(H,Y,X)'         , true , '_Get%1Field2(H,%B,Y,X)'    , va{:});

end

%***********************************************************************************************************************

function WriteFields(q, cat, sel, sym, cast, rep, varargin)

for i = 1 : numel(q.d.cat.(cat).syms)

    sname = q.d.cat.(cat).syms{i};
    sd = q.d.sym.(sname);

    switch q.ref
    case 't'
        % Don't process if the field was inherited.
        if sd.typeNo ~= q.typeNo, continue; end
    case 'd'
        % Same as 't', except we delay processing until we reach the subtype that defines dimensionality.
        if q.d.dimTypeNo == 0, continue; end
        if (q.d.dimTypeNo ~= q.typeNo) && (sd.typeNo ~= q.typeNo), continue; end
    end

    if any(sel == 's') &&  sd.multi  , continue; end
    if any(sel == 'm') && ~sd.multi  , continue; end
    if any(sel == 'p') &&  sd.private, continue; end

    if isempty(strfind(sym, '%C'))
        WriteField(q, sname, sd, sym, cast, rep, varargin{:});
    else
        WriteCoords(q, sname, sd, sym, cast, rep, varargin{:});
    end

end

end

%***********************************************************************************************************************

function WriteCoords(q, sname, sd, symFormat, cast, repFormat, varargin)

if isfield(sd, 'dims')
    d = sd;
else
    d = q.d;
end

for i = 1 : numel(d.dims)

    if isempty(d.dnames{i}), continue; end

    for j = 1 : numel(d.dims{i})

        if isscalar(d.dims{i})
            dimName = d.dnames{i};
        else
            dimName = sprintf('%s%u', d.dnames{i}, j - 1);
        end

        dimNo = sum(cellfun(@numel, d.dims(1 : i - 1))) + j;

        sym = symFormat;
        rep = repFormat;

        if ~isempty(strfind(sym, '%C'))
            sym = strrep(sym, '%C', upper(dimName));
        end

        if ~isempty(strfind(rep, '%D'))
            rep = strrep(rep, '%D', sprintf('%u', dimNo - 1));
        end

        WriteField(q, sname, sd, sym, cast, rep, varargin{:});

    end

end
    
end

%***********************************************************************************************************************

function WriteField(q, sname, sd, sym, cast, rep, varargin)

if isfield(sd, 'dims')
    d = sd;
else
    d = q.d;
end

if ~isempty(strfind(sym, '%H'))
    if q.dc.isType(q.typeNo), return; end
    sym = strrep(sym, '%H', [upper(q.names{q.typeNo}) '_' upper(sname)]);
end

if ~isempty(strfind(sym, '%I'))
    if q.dc.isType(q.typeNo)
        sym = strrep(sym, '%I', upper(sname));
    else
        sym = strrep(sym, '%I', [upper(q.names{q.typeNo}) '_' upper(sname)]);
    end
end

if ~isempty(strfind(sym, '%T'))
    name = upper(q.names{q.typeNo});
    if strcmp(name, 'BASE'), name = 'LAYER'; end
    sym = strrep(sym, '%T', name);
end

if ~isempty(strfind(sym, '%N'))
    sym = strrep(sym, '%N', upper(sname));
end

if ~isempty(strfind(sym, '%X'))
    sym = strrep(sym, '%X', DimList(d));
end

for i = 1 : numel(varargin)
    if ~isempty(strfind(sym, sprintf('%%%u', i)))
        if isnumeric(varargin{i})
            sym = strrep(sym, sprintf('%%%u', i), int2str(varargin{i}));
        else
            sym = strrep(sym, sprintf('%%%u', i), varargin{i});
        end
    end
end

if q.on

    if ~isempty(strfind(rep, '%U'))
        rep = strrep(rep, '%U', upper(q.names{q.typeNo}));
    end

    if ~isempty(strfind(rep, '%N'))
        rep = strrep(rep, '%N', upper(sname));
    end

    if ~isempty(strfind(rep, '%P'))
        rep = strrep(rep, '%P', upper(q.names{sd.ptrTypeNo}));
    end

    if ~isempty(strfind(rep, '%Z'))
        rep = strrep(rep, '%Z', sprintf('_Z_%s(Z)', upper(q.names{q.typeNo})));
    end

    if ~isempty(strfind(rep, '%F'))
        rep = strrep(rep, '%F', sprintf('%u', sd.pos - 1));
    end

    if ~isempty(strfind(rep, '%M'))
        rep = strrep(rep, '%M', sprintf('%u', 2 * (sd.pos - 1)));
    end

    if ~isempty(strfind(rep, '%B'))
        rep = strrep(rep, '%B', sprintf('%us%u', sd.resNo - 1, q.d.dimNo - 1));
    end

    if ~isempty(strfind(rep, '%R'))
        rep = strrep(rep, '%R', sprintf('%u', sd.resNo - 1));
    end

    if ~isempty(strfind(rep, '%S'))
        rep = strrep(rep, '%S', sprintf('%u', q.d.dimNo - 1));
    end

    if ~isempty(strfind(rep, '%A'))
        rep = strrep(rep, '%A', '__VA_ARGS__');
    end

    if ~isempty(strfind(rep, '%V'))
        if sd.int < 0
            rep = strrep(rep, '%V', '__int_as_float(V)');
        else
            rep = strrep(rep, '%V', 'V');
        end
    end

    if ~isempty(strfind(rep, '%X'))
        rep = strrep(rep, '%X', DimList(d));
    end

    for i = 1 : numel(varargin)
        if ~isempty(strfind(rep, sprintf('%%%u', i)))
            if isnumeric(varargin{i})
                rep = strrep(rep, sprintf('%%%u', i), int2str(varargin{i}));
            else
                rep = strrep(rep, sprintf('%%%u', i), varargin{i});
            end
        end
    end

    if cast && (sd.int < 0)
        rep = sprintf('__float_as_int(%s)', rep);
    end

    printf(q.refNo, SymNum(q.d, sname), '#define %s %s\n', sym, rep);

else

    i = find(sym == '(', 1);
    if ~isempty(i), sym = sym(1 : i - 1); end

    printf('#undef %s\n', sym);

end

end

%***********************************************************************************************************************

function s = DimList(d)

k = 0;
for i = 1 : numel(d.dims)
    for j = 1 : numel(d.dims{i})
        k = k + 1;
        if k == 1
            s = sprintf('C%u', k);
        else
            s = sprintf('%s,C%u', s, k);
        end
    end
end

end

%***********************************************************************************************************************

function WriteDef(q, sym, rep)

if q.on

    printf(q.refNo, 0, '#define %s %s\n', sym, rep);

else

    i = find(sym == '(', 1);
    if ~isempty(i), sym = sym(1 : i - 1); end

    printf('#undef %s\n', sym);

end

end

%***********************************************************************************************************************

function WriteConsts(on, d)

% TODO: should probably do something with refTypeNos.

for i = 1 : numel(d.cat.c.syms)

    name = d.cat.c.syms{i};

    value = d.sym.(name).value;
    if numel(value) ~= 1, continue; end

    if on

        switch d.sym.(name).int
        case  0, rep = sprintf('%ff', value);
        case -1, rep = sprintf('%i' , value);
        case -2, rep = sprintf('%i' , value - 1);
        end

        printf(0, SymNum(d, name), '#define %s %s\n', upper(name), rep);

    else

        printf('#undef %s\n', upper(name));

    end

end

end

%***********************************************************************************************************************

function GenerateKernelRun(def, platform)

for i = 1 : numel(def.types)

    name = def.types{i};
    d = def.type.(name);
    if ~d.kernel, continue; end

    a = ArrayInfo(d, platform);

    printf('#define _USER_KERNEL_TYPENO %u\n', d.typeNo - 1);
    printf('#define _USER_KERNEL_NAME _kernel_%s\n', name);
    printf('#define _USER_KERNEL_ARRAYBYTES %u\n', a.totalBytes);

    printf('#include "%s"\n', fullfile(SourcePath, 'kernel_run.h'));

    printf('#undef _USER_KERNEL_TYPENO\n');
    printf('#undef _USER_KERNEL_NAME\n');
    printf('#undef _USER_KERNEL_ARRAYBYTES\n');

end

end

%***********************************************************************************************************************

function a = ArrayInfo(d, platform)

a.names = d.cat.a.syms;

elements = zeros(1, numel(a.names));
intFlags = zeros(1, numel(a.names));
a.macros = cell (1, numel(a.names));
bytes    = zeros(1, numel(a.names));

for i = 1 : numel(a.names)

    elements(i) = d.sym.(a.names{i}).size;
    intFlags(i) = d.sym.(a.names{i}).int;

    switch intFlags(i)
    case -1, a.macros{i} = '_Int32ArrayElement' ; bytes(i) = 4;
    case  0, a.macros{i} = '_FloatArrayElement' ; bytes(i) = 4;
    case  3, a.macros{i} = '_DoubleArrayElement'; bytes(i) = 8;
    end

end

[ans, inds] = sort(bytes, 'descend');
a.names  = a.names (inds);
elements = elements(inds);
intFlags = intFlags(inds);
a.macros = a.macros(inds);
bytes    = bytes   (inds);

sizes = elements .* bytes;

a.offsets = zeros(1, numel(a.names));

for i = 1 : numel(a.names)

    switch platform
    case 'cuda', a.offsets(i) = sum(sizes(1 : i - 1)) / bytes(i);
    case 'cpu' , a.offsets(i) = sum(elements(intFlags(1 : i - 1) == intFlags(i)));
    end

end

a.int32Size  = sum(elements(intFlags == -1)); % These are only used for CPU.
a.floatSize  = sum(elements(intFlags ==  0));
a.doubleSize = sum(elements(intFlags ==  3));

a.totalBytes = sum(sizes); % This is only used for CUDA.

end

%***********************************************************************************************************************

function printf(varargin)

args = varargin;

if isnumeric(args{1})
    type = args{1};
    num  = args{2};
    args = args(3 : end);
else
    type = 0;
    num  = 0;
end

line = sprintf(args{1}, args{2 : end});

if HelpMode

    if ~isempty(regexp(line, '^#define [^_].*', 'once'))
        [ans, rest] = strtok(line, ' ');
        sym = strtok(rest, ' ');
        Help.type(end + 1) = type;
        Help.num (end + 1) = num;
        Help.sub (end + 1) = sum((Help.type == type) & (Help.num == num));
        Help.sym {end + 1} = sym;
    end

else

    fprintf(F, '%s', line);

end

end

%***********************************************************************************************************************

function num = SymNum(d, name)

if isempty(name)
    num = 0;
else
    num = find(strcmp(d.syms, name));
    if ~isscalar(num), error('unable to find symbol "%s"', name); end
end

end

%***********************************************************************************************************************

function Compile(platform, option, varargin)

inputFilePath = UserFilePath(false, '%s.%s', platform, SrcExt(platform));

outputFilePaths = {};
for i = 1 : numel(varargin)
    outputFilePaths{i} = UserFilePath(i == numel(varargin), varargin{i}, platform);
end

path = cd(UserPath);
[ans, output] = cns_compile(platform, option, inputFilePath, outputFilePaths{:});
cd(path);

Display(output, platform);

end

%***********************************************************************************************************************

function Display(output, platform)

suppress = {};
switch platform
case 'cuda'
    suppress{end + 1} = 'warning: variable "_.*" was declared but never referenced';
    suppress{end + 1} = 'Cannot tell what pointer points to';
case 'cpu'
    suppress{end + 1} = 'warning C4101: ''_.*'' : unreferenced local variable';
end

lines = strread(output, '%s', 'delimiter', sprintf('\n'), 'whitespace', '');

for i = 1 : numel(lines)

    disp = ~isempty(lines{i});

    if disp
        for j = 1 : numel(suppress)
            if ~isempty(regexp(lines{i}, suppress{j}, 'once'))
                disp = false;
                break;
            end
        end
    end

    if disp
        fprintf('%s\n', lines{i});
    end

end

end

%***********************************************************************************************************************

function ext = SrcExt(platform)

switch platform
case 'cuda', ext = 'cu';
case 'cpu' , ext = 'cpp';
end

end

%***********************************************************************************************************************

function Cleanup

if F >= 0
    fclose(F);
    F = -1;
end

if ~KeepGenerated
    list = dir(fullfile(UserPath, [Package '_cns_generated_*']));
    for i = 1 : numel(list)
        delete(fullfile(UserPath, list(i).name));
    end
end

end

%***********************************************************************************************************************

function DeleteMex(platform)

func = [Package '_cns_compiled_' platform];
filePath = fullfile(UserPath, [func '.' mexext]);

if mislocked(func)
    tempPath = which(func);
    if isempty(tempPath)
        error('the "%s" function is locked but not in the path; try restarting MATLAB', func);
    end
    if ~exist(tempPath, 'file')
        error('the "%s" function is locked but has been deleted; try restarting MATLAB', func);
    end
    feval(func, -1);
end
clear(func);
if mislocked(func)
    error('the "%s" function is currently locked; try restarting MATLAB', func);
end

if exist(filePath, 'file')
    delete(filePath);
    if exist(filePath, 'file')
        error('the "%s" function is currently locked; try restarting MATLAB', func);
    end
end

end

%***********************************************************************************************************************

function DeleteFile(fileName, varargin)

filePath = UserFilePath(true, fileName, varargin{:});

if exist(filePath, 'file')
    delete(filePath);
end

end

%***********************************************************************************************************************

function filePath = UserFilePath(perm, fileName, varargin)

if perm
    prefix = 'compiled_';
else
    prefix = 'generated_';
end

filePath = fullfile(UserPath, [Package '_cns_' prefix sprintf(fileName, varargin{:})]);

end
        
%***********************************************************************************************************************

end
