@echo off
setlocal
cd /d "%~dp0"

echo ==========================================
echo Walmart AI Demand Forecasting
echo ==========================================

if not exist "venv\Scripts\python.exe" (
    echo ERROR: Virtual environment not found.
    echo Create it with: python -m venv venv
    echo Then install requirements with:
    echo venv\Scripts\python.exe -m pip install -r requirements.txt
    pause
    exit /b 1
)

echo.
echo [1/3] Preparing data...
venv\Scripts\python.exe src\prepare_data.py
if errorlevel 1 goto error

echo.
echo [2/3] Training model...
venv\Scripts\python.exe src\train_model.py
if errorlevel 1 goto error

echo.
echo [3/3] Starting Streamlit dashboard...
venv\Scripts\python.exe -m streamlit run app.py
if errorlevel 1 goto error
goto end

:error
echo.
echo ==========================================
echo A project step failed. Read the error above.
echo ==========================================
pause
exit /b 1

:end
endlocal
