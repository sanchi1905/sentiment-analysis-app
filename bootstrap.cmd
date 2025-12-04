@echo off
REM bootstrap.cmd - reset and bootstrap using the safe clean workflow

echo Resetting project (will remove the `venv` directory and overwrite `requirements.txt`)...

if exist venv (
    echo Removing existing venv...
    rmdir /s /q venv
)

echo Copying clean requirements into place...
copy /Y requirements.clean.txt requirements.txt >nul
if %ERRORLEVEL% NEQ 0 (
    echo Failed to copy requirements.clean.txt to requirements.txt
    exit /b 1
)

echo Creating fresh virtual environment and installing dependencies...
call create_venv.cmd
if %ERRORLEVEL% NEQ 0 (
    echo create_venv.cmd failed. Check output above.
    exit /b %ERRORLEVEL%
)

echo Running training script to generate model pipeline (this may take a while)...
venv\Scripts\python train.py
if %ERRORLEVEL% NEQ 0 (
    echo Training failed. Check the output above.
    exit /b %ERRORLEVEL%
)

echo Reset and bootstrap complete.
echo To activate the venv: venv\Scripts\activate
echo To run the app: streamlit run app.py
