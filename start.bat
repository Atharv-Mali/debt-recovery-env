@echo off
echo ========================================================
echo DebtRecovery Environment - Startup
echo ========================================================
echo.
echo [1/2] Compiling Dashboard HTML...
python dashboard\build_dashboard.py
echo.
echo [2/2] Starting API Backend on http://0.0.0.0:7860...
echo.
echo View Dashboard at: http://localhost:7860/dashboard
echo ========================================================
python -m uvicorn app:app --host 0.0.0.0 --port 7860
