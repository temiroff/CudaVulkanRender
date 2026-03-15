@echo off
echo === CudaPathTracer Setup ===

REM Clone submodules
if not exist "external\imgui" (
    echo Cloning ImGui (docking branch)...
    git submodule add -b docking https://github.com/ocornut/imgui external/imgui
)
if not exist "external\glfw" (
    echo Cloning GLFW...
    git submodule add https://github.com/glfw/glfw external/glfw
)

git submodule update --init --recursive

REM Create build dir
if not exist "build" mkdir build

echo.
echo === Configuring CMake ===
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
if errorlevel 1 (
    echo CMake configure failed. Check that Vulkan SDK and CUDA Toolkit are installed.
    pause
    exit /b 1
)

echo.
echo Setup complete!
echo To build: cmake --build build --config Release
echo To run:   build\Release\pathtracer.exe
pause
