@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 (
    echo Failed to initialize VS environment
    exit /b 1
)
set SRC=%~dp0
set SRC=%SRC:~0,-1%
cmake -S "%SRC%" -B "%SRC%\build" -G "Visual Studio 17 2022" -A x64
