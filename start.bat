@echo off
cd /d "%~dp0"
echo Running from: %cd%
call venv\Scripts\activate
python app.py
pause
