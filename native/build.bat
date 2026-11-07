@echo off
setlocal enabledelayedexpansion

:: Try to initialize MSVC Developer environment if cl is not already on PATH
where cl.exe >nul 2>&1
if errorlevel 1 (
    if exist "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" (
        call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
    ) else if exist "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" (
        call "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
    ) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" (
        call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
    ) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" (
        call "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
    ) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat" (
        call "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
    )
)

where cl.exe >nul 2>&1
if errorlevel 1 (
    echo Error: cl.exe compiler not found. Please run from Visual Studio Developer Command Prompt or install MSVC C++ tools.
    exit /b 1
)

pushd "%~dp0.."
if not exist build mkdir build

echo Compiling native\threader.cpp ...
cl /O2 /std:c++17 /EHsc /utf-8 /I native native\threader.cpp /Fo:build\ /Fe:build\threader.exe
if errorlevel 1 (
    echo Compilation failed.
    popd
    exit /b 1
)

popd
echo Build complete: build\threader.exe
