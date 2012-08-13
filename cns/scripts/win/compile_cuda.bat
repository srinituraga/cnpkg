@echo off

rem 1 = mex include path
rem 2 = setup filepath
rem 3 = input filepath
rem 4 = option
rem 5 = intermediate filepath
rem 6 = output filepath

call %2

if not "%CUDA%"=="1" exit 0

if not exist %3 (
    echo can't find %3
    exit 1
)

%SETUP_COMPILER%

if "%4"=="compile" (
    nvcc -cuda -I %1 %NVCC_OPTIONS% -use_fast_math -o %5 %3
) else if "%4"=="preprocess" (
    nvcc -E -I %1 %NVCC_OPTIONS% -use_fast_math -o %5 %3
) else if "%4"=="info" (
    nvcc -cubin -I %1 %NVCC_OPTIONS% -use_fast_math -o %5 %3
)

if exist %5 (
    echo CUDA preprocessing successful
) else (
    echo CUDA preprocessing [nvcc] failed
    exit 1
)

if "%4"=="compile" (
    call mex -output %6 %5 %CUDA_LINK_LIB%
) else (
    exit 0
)

if exist %6 (
    echo CUDA compilation successful
    exit 0
) else (
    echo CUDA compilation [mex] failed
    exit 1
)
