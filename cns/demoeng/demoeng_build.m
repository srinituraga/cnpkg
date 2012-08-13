function demoeng_build

path = fileparts(mfilename('fullpath'));
name = 'demoeng.cpp';

switch lower(computer)
case 'pcwin'  , args = {'-f', [matlabroot '\bin\win32\mexopts\msvc80engmatopts.bat']};
case 'pcwin64', args = {'-f', [matlabroot '\bin\win64\mexopts\msvc80engmatopts.bat']};
otherwise     , args = {'-f', [matlabroot '/bin/engopts.sh']};
end

oldpath = cd(path);

fprintf('\nMEX IS EXECUTING THESE COMMANDS:\n\n');

try
    mex('-n', args{:}, name);
    mex(args{:}, name);
    err = [];
catch
    err = lasterror;
end

cd(oldpath);

if ~isempty(err), rethrow(err); end

return;