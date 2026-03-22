@echo off
setlocal
if "%~1"=="" (
    echo Usage: highlight.bat ^<pdf-file^>
    echo Example: highlight.bat paper.pdf
    exit /b 1
)
set "INPUT=%~1"
set "OUTPUT=%~dp1%~n1_highlighted.pdf"
if exist "%~dp0.venv\Scripts\python.exe" (
    "%~dp0.venv\Scripts\python.exe" -m pdf_highlighter "%INPUT%" -o "%OUTPUT%" --provider gemini
) else (
    python -m pdf_highlighter "%INPUT%" -o "%OUTPUT%" --provider gemini
)
pause
exit /b %ERRORLEVEL%
