@echo off
REM ==== TESR YOLO Trainer - double-click to train (Windows) ====
REM First run: makes its own Python environment (.venv) and installs everything (internet needed).
REM Next runs: trains right away. Change the 3 settings below if you like.
setlocal
cd /d "%~dp0"
set "NAME=__NAME__"
set "IMGSZ=__IMGSZ__"
set "EPOCHS=__EPOCHS__"
title TESR YOLO Trainer - %NAME%
echo ==================================================
echo   TESR YOLO Trainer __VERSION__  -  %NAME%
echo   image size %IMGSZ%  -  %EPOCHS% epochs
echo ==================================================

REM ---- 1. find Python 3.10 - 3.14 ----
set "PY="
for %%V in (3.12 3.11 3.13 3.10 3.14) do call :trypy %%V
if defined PY goto havepy
python -c "import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,14) else 1)" >nul 2>nul
if not errorlevel 1 set "PY=python"
:havepy
if not defined PY goto nopython
echo Python: %PY%

REM ---- 2. own environment, made once ----
if exist ".venv\Scripts\python.exe" goto haveenv
echo.
echo [1/3] Making the Python environment .venv ...
%PY% -m venv .venv || goto err
:haveenv
set "VPY=.venv\Scripts\python.exe"
if exist ".venv\tesr_ok.txt" goto train
echo.
echo [2/3] Installing YOLO - first time only, takes a few minutes ...
"%VPY%" -m pip install --upgrade pip
where nvidia-smi >nul 2>nul
if not errorlevel 1 goto gpu
echo No NVIDIA card found - training on the CPU (slower, works fine for small datasets).
goto deps
:gpu
echo NVIDIA card found - installing PyTorch with CUDA (about 2.5 GB) ...
"%VPY%" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126 || echo CUDA PyTorch failed - continuing on the CPU.
:deps
"%VPY%" -m pip install -r requirements.txt || goto err
echo ok> ".venv\tesr_ok.txt"

REM ---- 3. train + export NAME.onnx ----
:train
echo.
echo [3/3] Training %NAME% ...
"%VPY%" train.py --imgsz %IMGSZ% --epochs %EPOCHS% --onnx %NAME% || goto err
if not exist "%NAME%.onnx" goto err
echo.
echo ==================================================
echo   Done!  %NAME%.onnx  is in this folder.
echo   ROI Studio: + Add a trained model - pick %NAME%.onnx
echo   Test with the webcam:  .venv\Scripts\python run.py
echo   1 zip = 1 model. Two AIs (find + count)? Download the other project too.
echo ==================================================
start "" explorer /select,"%~dp0%NAME%.onnx"
pause
exit /b 0

:trypy
if defined PY exit /b 0
py -%1 -c "1" >nul 2>nul
if not errorlevel 1 set "PY=py -%1"
exit /b 0

:nopython
echo.
echo Python 3.10 - 3.14 was not found.
echo Install it from python.org - tick "Add python.exe to PATH" - then double-click this file again.
start "" https://www.python.org/downloads/
pause
exit /b 1

:err
echo.
echo Something went wrong - read the red lines above. Common fixes:
echo  - no internet on the first run: connect and run again
echo  - out of memory: open this file in Notepad and set IMGSZ=320
echo  - start clean: delete the .venv folder and run again
pause
exit /b 1
