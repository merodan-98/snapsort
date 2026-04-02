@echo off
echo ===========================================
echo SnapSort MVP - Desktop GUI Launcher
echo ===========================================
echo Loading Machine Learning Models...
echo Please wait a few seconds...

python 4_gui_app.py
if %errorlevel% neq 0 (
    echo.
    echo [WARNING] System 'python' not found.
    echo Trying fallback local installation path...
    "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" 4_gui_app.py
)

echo.
echo Program Terminated.
pause
