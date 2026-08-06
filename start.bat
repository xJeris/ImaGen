@echo off
cd /d "%~dp0"
echo Running from: %cd%
echo Loading...
call venv\Scripts\activate
python server.py
set EC=%errorlevel%
if %EC% equ 0 exit
if %EC% equ 42 exit
if %EC% equ 15 exit
if %EC% equ -15 exit
if %EC% equ 3221225725 exit
echo.
echo ImaGen exited with an error (code %EC%).
pause
