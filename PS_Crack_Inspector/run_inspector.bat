@echo off
cd /d "%~dp0"
python ps_app.py
if errorlevel 1 pause
