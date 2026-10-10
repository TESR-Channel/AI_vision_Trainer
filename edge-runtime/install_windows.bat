@echo off
REM ==== TESR web model - one-click installer (Windows) ====
setlocal
set "PY=python"
py -3 -c "print()" >nul 2>nul && set "PY=py -3"
echo Using: %PY%
%PY% -m venv venv || goto err
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt || goto err
echo.
echo ============================================
echo  Done! To run your model:
echo    venv\Scripts\activate
echo    python predict.py
echo ============================================
pause
exit /b 0
:err
echo.
echo Install failed - see README_DEPLOY.md (needs Python 3.8 or newer from python.org).
pause
exit /b 1
