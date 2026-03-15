@echo off
cd /d "%~dp0"
echo Running from: %cd%
echo Loading...
call venv\Scripts\activate
python app.py
set EC=%errorlevel%
if %EC% equ 0 exit
if %EC% equ 15 exit
if %EC% equ -15 exit
if %EC% equ 3221225725 exit
echo.
echo ImaGen exited with an error (code %EC%).
pause
