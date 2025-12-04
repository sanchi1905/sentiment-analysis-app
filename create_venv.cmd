@echo off
REM create_venv.cmd - create a fresh venv and install project requirements
if exist venv (
    echo A 'venv' directory already exists. Remove it first if you want a fresh environment.
    goto :install
)

echo Creating virtual environment 'venv'...
python -m venv venv
if %ERRORLEVEL% NEQ 0 (
    echo Failed to create virtualenv. Ensure Python is on PATH.
    exit /b 1
)

:install
echo Upgrading pip, setuptools and wheel inside the venv...
venv\Scripts\python -m pip install --upgrade pip setuptools wheel
echo Installing requirements from requirements.clean.txt...
venv\Scripts\pip install -r requirements.clean.txt
if %ERRORLEVEL% NEQ 0 (
    echo pip install failed. Check the output above for errors.
    exit /b %ERRORLEVEL%
)

echo Installing NLTK data (stopwords/punkt)...
venv\Scripts\python ensure_nltk_data.py

echo Virtual environment ready. Activate it with:
echo     venv\Scripts\activate
