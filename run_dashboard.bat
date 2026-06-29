@echo off
echo ============================================
echo   מערכת זיהוי דפוסי למידה - Dashboard
echo ============================================
echo.

cd /d "%~dp0"

set PY=C:\Users\saed\AppData\Local\Programs\Python\Python312\python.exe
set PIP=C:\Users\saed\AppData\Local\Programs\Python\Python312\Scripts\pip.exe

echo Checking Python...
"%PY%" --version

echo Installing streamlit if needed...
"%PIP%" install streamlit plotly -q

echo.
echo Generating data if needed...
"%PY%" generate_data.py

echo.
echo Starting Dashboard...
echo Open your browser at: http://localhost:8501
echo (Press CTRL+C to stop)
echo.
"%PY%" -m streamlit run dashboard.py --server.address 0.0.0.0 --server.port 8501
pause
