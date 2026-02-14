@echo off
REM ============================================================
REM TitanInfer Build Script (Windows)
REM ============================================================

setlocal enabledelayedexpansion

set PROJECT_DIR=%~dp0..
set BUILD_DIR=%PROJECT_DIR%\build
set BUILD_TYPE=Release
set RUN_TESTS=false
set ENABLE_SIMD=ON

:parse_args
if "%~1"=="" goto :done_args
if /i "%~1"=="/debug"   (set BUILD_TYPE=Debug& shift& goto :parse_args)
if /i "%~1"=="/clean"   (if exist "%BUILD_DIR%" rmdir /s /q "%BUILD_DIR%"& shift& goto :parse_args)
if /i "%~1"=="/test"    (set RUN_TESTS=true& shift& goto :parse_args)
if /i "%~1"=="/no-simd" (set ENABLE_SIMD=OFF& shift& goto :parse_args)
if /i "%~1"=="/help"    (goto :usage)
echo Unknown option: %~1
goto :usage
:done_args

if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"

echo Build type: %BUILD_TYPE%
echo SIMD: %ENABLE_SIMD%
echo.

REM Configure
cmake -S "%PROJECT_DIR%" -B "%BUILD_DIR%" ^
    -DCMAKE_BUILD_TYPE=%BUILD_TYPE% ^
    -DENABLE_SIMD=%ENABLE_SIMD%
if errorlevel 1 goto :error

REM Build
cmake --build "%BUILD_DIR%" --config %BUILD_TYPE%
if errorlevel 1 goto :error

echo.
echo Build completed successfully.

REM Optionally run tests
if "%RUN_TESTS%"=="true" (
    echo.
    echo Running tests...
    cd "%BUILD_DIR%" && ctest --output-on-failure -C %BUILD_TYPE%
    if errorlevel 1 goto :error
)

goto :eof

:usage
echo Usage: %~nx0 [options]
echo.
echo Options:
echo   /debug      Build in Debug mode (enables bounds checking)
echo   /clean      Remove build directory before building
echo   /test       Run tests after building
echo   /no-simd    Disable AVX2/FMA SIMD optimizations
echo   /help       Show this help message
goto :eof

:error
echo.
echo Build failed!
exit /b 1
