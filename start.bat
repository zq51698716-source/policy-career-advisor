@echo off
title Policy Career Advisor
cd /d "%~dp0"

REM Find Python
set PY=
for %%d in ("%LOCALAPPDATA%\Programs\Python\Python311" "%LOCALAPPDATA%\Programs\Python\Python312" "%LOCALAPPDATA%\Programs\Python\Python313" "%LOCALAPPDATA%\Programs\Python\Python314" "%PROGRAMFILES%\Python312") do (
    if exist "%%~d\python.exe" set "PY=%%~d\python.exe"
)
if "%PY%"=="" (
    echo Python not found!
    pause
    exit /b 1
)
echo Python: %PY%

REM Check .env
if not exist ".env" (
    echo Creating .env from template...
    copy .env.example .env >nul
    echo Please edit .env and add your API Keys, then re-run.
    start notepad .env
    pause
    exit /b 1
)

REM Install deps if needed
%PY% -c "import fastapi" 2>nul
if %errorlevel% neq 0 (
    echo Installing dependencies...
    %PY% -m pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo Install failed! Try manually: %PY% -m pip install -r requirements.txt
        pause
        exit /b 1
    )
)

REM Start
echo Starting server at http://localhost:8000
echo Press Ctrl+C to stop
%PY% -m uvicorn backend.main:app --reload --port 8000 --host 0.0.0.0
pause
