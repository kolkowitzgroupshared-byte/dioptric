@echo off

REM Set your dioptric repo path
set "DIOPTRIC_DIR=%USERPROFILE%\Github\dioptric"
set "LABRAD_MANAGER=%DIOPTRIC_DIR%\labradmanager"

REM Start LabRAD server
start "LabRAD" cmd /k ""%LABRAD_MANAGER%\scalabrad-0.8.3\bin\labrad.bat""

REM Start LabRAD web server
start "LabRAD Web" cmd /k ""%LABRAD_MANAGER%\scalabrad-web-server-2.0.5\bin\labrad-web.bat""

REM Open LabRAD web interface
start "" chrome --new-window "http://localhost:7667"

REM Activate conda environment
if exist "%USERPROFILE%\miniconda3\condabin\conda.bat" (
    call "%USERPROFILE%\miniconda3\condabin\conda.bat" activate dioptric
) else if exist "C:\ProgramData\miniconda3\condabin\conda.bat" (
    call "C:\ProgramData\miniconda3\condabin\conda.bat" activate dioptric
) else (
    echo Could not find conda.bat
    pause
    exit /b 1
)

REM Start LabRAD node
python -m labrad.node -u "" -w ""

pause
