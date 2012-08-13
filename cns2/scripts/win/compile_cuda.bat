@echo off

rem 1 = mex script filepath
rem 2 = mex include path
rem 3 = script path (containing this file)
rem 4 = input filepath
rem 5 = option
rem 6 = intermediate filepath
rem 7 = output filepath

call %3\setup.bat

if not "%CUDA%"=="1" exit 0

if not exist %4 (
    echo can't find %4
    exit 1
)

%SETUP_COMPILER%

if "%5"=="compile" (
    nvcc -cuda -I %2 %NVCC_OPTIONS% -use_fast_math -o %6 %4
) else if "%5"=="preprocess" (
    nvcc -E -I %2 %NVCC_OPTIONS% -use_fast_math -o %6 %4
) else if "%5"=="info" (
    nvcc -cubin -I %2 %NVCC_OPTIONS% -use_fast_math -o %6 %4
)

if exist %6 (
    echo CUDA preprocessing successful
) else (
    echo CUDA preprocessing [nvcc] failed
    exit 1
)

if "%5"=="compile" (
    call %1 -output %7 %6 %CUDA_LINK_LIB%
) else (
    exit 0
)

if exist %7 (
    echo CUDA compilation successful
    exit 0
) else (
    echo CUDA compilation [mex] failed
    exit 1
)
