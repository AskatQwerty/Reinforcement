@echo off
setlocal
cd /d "%~dp0"
title Night Shift V2
if exist ".venv\Scripts\python.exe" goto check_dependency
where python >nul 2>nul
if errorlevel 1 goto try_py
python -m venv .venv
if errorlevel 1 goto failed
goto check_dependency
:try_py
where py >nul 2>nul
if errorlevel 1 goto no_python
py -3 -m venv .venv
if errorlevel 1 goto failed
:check_dependency
".venv\Scripts\python.exe" -c "import pygame" >nul 2>nul
if not errorlevel 1 goto play
echo Installing game dependency. Internet is needed on the first launch only.
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r requirements.txt
if errorlevel 1 goto failed
:play
".venv\Scripts\python.exe" main.py
if errorlevel 1 goto failed
exit /b 0
:no_python
echo Python 3.10 or newer is required. Install Python from https://www.python.org/
echo Enable "Add Python to PATH" during installation, then run this file again.
pause
exit /b 1
:failed
echo.
echo The game could not start. The error message is shown above.
pause
exit /b 1
