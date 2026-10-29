@echo off
title Ender 3 Cutter Studio Web Server
cd /d "%~dp0"

echo ======================================================
echo       Ender 3 Cutter Studio Web Server Launcher
echo ======================================================

set "PYTHON_PATH=C:\Users\chait\AppData\Local\Programs\Python\Python39"
if exist "%PYTHON_PATH%\python.exe" (
    set "PATH=%PYTHON_PATH%;%PYTHON_PATH%\Scripts;%PATH%"
    set "PY_EXE=%PYTHON_PATH%\python.exe"
) else (
    set "PY_EXE=python"
)

echo [INFO] Python Interpreter: %PY_EXE%
"%PY_EXE%" -c "import cv2, numpy, svgelements; print('  [OK] Python 3.9 + OpenCV + NumPy + svgelements verified.')" 2>nul
if errorlevel 1 (
    echo   [WARN] Missing packages, installing from requirements.txt...
    "%PY_EXE%" -m pip install -r requirements.txt
)

echo.
echo ======================================================
echo Web GUI running at: http://localhost:5000
echo Close this window or press Ctrl+C to stop the server.
echo ======================================================
echo.

start "" "http://localhost:5000"
"%PY_EXE%" web_gui.py --port 5000
pause
