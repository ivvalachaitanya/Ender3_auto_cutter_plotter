@echo off
title Ender 3 Cutter Studio Web Server
cd /d "%~dp0"
echo ======================================================
echo Starting Ender 3 Cutter Studio Web Server...
echo Port: 5000
echo Open in your browser: http://localhost:5000
echo ======================================================
"C:\Users\chait\AppData\Local\Programs\Python\Python39\python.exe" web_gui.py --port 5000
pause
