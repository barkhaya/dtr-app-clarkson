@echo off
title DTR Dashboard Launcher
color 0A

echo ============================================================
echo   IEEE C57.91 Dynamic Transformer Rating Dashboard
echo ============================================================
echo.

:: ── Find Python ───────────────────────────────────────────────
echo [1/3] Detecting Python...
echo.
set PYTHON_CMD=

:: 1) Try 'python'
python --version >nul 2>&1
if not errorlevel 1 (
    set PYTHON_CMD=python
    goto :python_found
)

:: 2) Try 'python3'
python3 --version >nul 2>&1
if not errorlevel 1 (
    set PYTHON_CMD=python3
    goto :python_found
)

:: 3) Try Windows Launcher 'py' (covers all installed versions)
py --version >nul 2>&1
if not errorlevel 1 (
    set PYTHON_CMD=py
    goto :python_found
)

:: 4) Try 'py -3.9' specifically
py -3.9 --version >nul 2>&1
if not errorlevel 1 (
    set PYTHON_CMD=py -3.9
    goto :python_found
)

:: 5) Search common install locations
for %%d in (
    "%LOCALAPPDATA%\Programs\Python\Python39\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    "C:\Python39\python.exe"
    "C:\Python310\python.exe"
    "C:\Python311\python.exe"
    "C:\Python312\python.exe"
    "C:\ProgramData\Anaconda3\python.exe"
    "C:\ProgramData\miniconda3\python.exe"
    "%USERPROFILE%\Anaconda3\python.exe"
    "%USERPROFILE%\miniconda3\python.exe"
) do (
    if exist %%d (
        set PYTHON_CMD=%%d
        goto :python_found
    )
)

:: 6) Nothing found — install Python automatically
echo       [MISSING] Python not found anywhere. Attempting automatic install...
echo.

:: Try winget first (built into Windows 11)
winget --version >nul 2>&1
if not errorlevel 1 (
    echo       Installing Python 3.11 via winget...
    winget install --id Python.Python.3.11 --silent --accept-package-agreements --accept-source-agreements
    if not errorlevel 1 goto :refresh_path
    echo       [WARNING] winget install failed. Trying direct download...
)

:download_python
echo       Downloading Python 3.11 installer...
set PYTHON_URL=https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe
set PYTHON_INSTALLER=%TEMP%\python_installer.exe
powershell -Command "Invoke-WebRequest -Uri '%PYTHON_URL%' -OutFile '%PYTHON_INSTALLER%'"
if not exist "%PYTHON_INSTALLER%" (
    echo       [ERROR] Download failed. Install Python manually from https://www.python.org/downloads/
    pause
    exit /b 1
)
echo       Running installer silently (no admin required)...
"%PYTHON_INSTALLER%" /quiet InstallAllUsers=0 PrependPath=1 Include_pip=1
del "%PYTHON_INSTALLER%" >nul 2>&1

:refresh_path
for /f "tokens=*" %%i in ('powershell -Command "[System.Environment]::GetEnvironmentVariable('PATH','User')"') do set "PATH=%%i;%PATH%"
set PYTHON_CMD=python

:: Confirm install worked
python --version >nul 2>&1
if errorlevel 1 (
    echo       [ERROR] Installation failed.
    echo       Please install Python manually: https://www.python.org/downloads/
    echo       Tick "Add Python to PATH" during setup, then re-run this file.
    pause
    exit /b 1
)
echo       [OK] Python installed successfully!

:python_found
for /f "tokens=*" %%v in ('%PYTHON_CMD% --version 2^>^&1') do echo       Found: %%v
echo       Command: %PYTHON_CMD%
echo.

:: ── Check and install dependencies ────────────────────────────
echo [2/3] Checking and installing dependencies...
echo.

set PACKAGES=streamlit numpy pandas scipy plotly geopy requests

for %%p in (%PACKAGES%) do (
    %PYTHON_CMD% -c "import %%p" >nul 2>&1
    if errorlevel 1 (
        echo       [MISSING] %%p -- installing...
        %PYTHON_CMD% -m pip install %%p --quiet
        if errorlevel 1 (
            echo       [ERROR] Failed to install %%p
            pause
            exit /b 1
        )
        echo       [OK] %%p installed successfully
    ) else (
        echo       [OK] %%p already installed
    )
)

echo.
echo       All dependencies satisfied!
echo.

:: ── Launch Streamlit app ───────────────────────────────────────
echo [3/3] Launching DTR Dashboard...
echo.
echo       App will open at: http://localhost:8501
echo       Press Ctrl+C in this window to stop the app.
echo.
echo ============================================================
echo.

cd /d "%~dp0"
%PYTHON_CMD% -m streamlit run DTR_app.py

pause
