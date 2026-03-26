@echo off
cd /d "F:\PROJECTS\NVDIA\CudaVulkan\build\Release"

:: Build timestamp: YYYY-MM-DD_HH-MM-SS
set HH=%time:~0,2%
if "%HH:~0,1%"==" " set HH=0%HH:~1,1%
set TIMESTAMP=%date:~-4%-%date:~4,2%-%date:~7,2%_%HH%-%time:~3,2%-%time:~6,2%
set OUTFILE=F:\PROJECTS\NVDIA\CudaVulkan\profiles\profile_%TIMESTAMP%

mkdir "F:\PROJECTS\NVDIA\CudaVulkan\profiles" 2>nul

"C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.1.0\ncu.bat" --set full --launch-count 100 -o "%OUTFILE%" pathtracer.exe

echo.
echo Profile saved to %OUTFILE%.ncu-rep
pause
