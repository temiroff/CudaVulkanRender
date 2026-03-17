@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set SRC=%~dp0
set SRC=%SRC:~0,-1%
set EXE=%SRC%\build\Release\pathtracer.exe
cmake --build "%SRC%\build" --config Release
if errorlevel 1 (
    echo Build failed.
    pause
    exit /b 1
)
echo.
echo === Launching pathtracer ===
echo EXE: %EXE%
for %%I in ("%EXE%") do echo EXE timestamp: %%~tI
"%EXE%"
