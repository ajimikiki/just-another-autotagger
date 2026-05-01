@echo off
cd /d "%~dp0"

set PYTHON=%~dp0python\python.exe

if not exist "%PYTHON%" (
    echo Python not found.
    pause
    exit /b
)

echo [33mJust Another Autotagger[0m
echo.

set /p folder=Enter path to images folder: 
if "%folder%"=="" exit /b

set /p trigger=Enter trigger tag (or leave empty): 
echo.

if "%trigger%"=="" (
    "%PYTHON%" "%~dp0libs\autotag.py" --folder "%folder%"
) else (
    "%PYTHON%" "%~dp0libs\autotag.py" --folder "%folder%" --trigger "%trigger%"
)

echo.

pause