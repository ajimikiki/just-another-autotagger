@echo off
cd /d "%~dp0"
title Just Another Autotagger - Setup

set PYTHON=%~dp0python\python.exe

if not exist "%PYTHON%" (
    echo [ERROR] Python not found.
    pause
    exit /b
)

echo Setting up Just Another Autotagger
echo.

echo - Installing pip...
"%PYTHON%" "%~dp0python\get-pip.py" >nul
if errorlevel 1 goto error
echo   [33m[OK][0m

echo - Installing dependencies...
"%PYTHON%" -m pip install pillow numpy tqdm huggingface_hub onnxruntime --target "%~dp0libs" >nul
if errorlevel 1 goto error
echo   [33m[OK][0m

echo.
echo Setup complete.
pause
exit /b

:error
echo   [ERROR]
echo.
echo Setup failed. Please check your internet connection or try again.
pause
exit /b