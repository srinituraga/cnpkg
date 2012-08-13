@echo off

setlocal

set libpath=C:\Program Files\MATLAB\R2009a\bin\win32;C:\CUDA\bin

set cnsdir=Z:\jim\libs\cns

path = %path%;%libpath%
.\demoeng.exe %cnsdir%

endlocal
