@echo off
setlocal
cd /d "%~dp0"

set LOG=%~dp0all_pull_run.log
echo ==== KH560 pull-out batch start %date% %time% ==== > "%LOG%"

echo [1/5] Running 2 A/ps pull-out ...
python "%~dp02A\run_pull_2A.py" >> "%LOG%" 2>&1
if errorlevel 1 goto :error
echo [1/5] 2 A/ps done.

echo [2/5] Running 5 A/ps pull-out ...
python "%~dp05A\run_pull_5A.py" >> "%LOG%" 2>&1
if errorlevel 1 goto :error
echo [2/5] 5 A/ps done.

echo [3/5] Running 10 A/ps pull-out ...
python "%~dp010A\run_pull_10A.py" >> "%LOG%" 2>&1
if errorlevel 1 goto :error
echo [3/5] 10 A/ps done.

echo [4/5] Running 20 A/ps pull-out ...
python "%~dp020A\run_pull_20A.py" >> "%LOG%" 2>&1
if errorlevel 1 goto :error
echo [4/5] 20 A/ps done.

echo [5/5] Running 15 / 25 / 30 A/ps pull-out ...
python "%~dp015-25-30A\run_pull_15_25_30.py" >> "%LOG%" 2>&1
if errorlevel 1 goto :error
echo [5/5] 15 / 25 / 30 A/ps done.

echo ==== KH560 pull-out batch finished %date% %time% ==== >> "%LOG%"
echo.
echo All pull-out jobs completed.
echo Log written to: %LOG%
exit /b 0

:error
echo ==== KH560 pull-out batch FAILED at %date% %time% ==== >> "%LOG%"
echo.
echo ERROR: pull-out batch stopped.
echo Last output is in: %LOG%
exit /b 1
