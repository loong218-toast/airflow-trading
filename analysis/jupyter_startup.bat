@echo off
:: 1. Go to your user folder (where your notebooks are)
cd /d "C:\Users\Owner"

:: 2. Activate your specific environment
:: Note: Using the standard path for Anaconda activation
call "C:\ProgramData\anaconda3\Scripts\activate.bat" dev_env

:: 3. Start Jupyter Lab for remote access
jupyter lab --ip=0.0.0.0 --no-browser
pause