@echo off

rem 1 = mex script filepath
rem 2 = mex include path
rem 3 = script path (containing this file)
rem 4 = input filepath
rem 5 = option
rem 6 = output filepath

call %3\setup.bat

if not exist %4 (
    echo can't find %4
    exit 1
)

if "%5"=="compile" (
    call %1 -output %6 %4
) else if "%5"=="preprocess" (
    %SETUP_COMPILER%
    cl -E -I %2 %4 > %6
)

if exist %6 (
    echo CPU compilation successful
    exit 0
) else (
    echo CPU compilation failed
    exit 1
)
