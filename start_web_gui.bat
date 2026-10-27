@echo off
REM Start Cutter Studio Web Server using Python 3.9
title Ender 3 Cutter Studio

echo ======================================================
echo Starting Ender 3 Cutter Studio Web Server...
echo ======================================================

set PY_EXE=C:\Users\chait\AppData\Local\Programs\Python\Python39\python.exe

if exist "%PY_EXE%" (
    "%PY_EXE%" web_gui.py --port 5000
) else (
    py -3.9 web_gui.py --port 5000
)

pause
