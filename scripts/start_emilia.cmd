@echo off
rem Emilia-YODAS JA (CC BY 4.0) replacement set for the viability run: ingest ~300 h, teacher pass, free second
rem opinion. Every step is resumable, so this is safe to re-run after a crash or reboot.
cd /d D:\Shizu-ko-distill
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set PY=C:\Users\multy\AppData\Local\Programs\Python\Python312\python.exe
echo %date% %time% START >> emilia_pipeline.log
"%PY%" scripts\01_prepare_data.py --sources emilia_yodas --emilia-hours 300 >> emilia_prepare.log 2>&1 || goto :fail
echo %date% %time% ingest done >> emilia_pipeline.log
"%PY%" scripts\run_teacher_pass.py --watch-dir teacher_out\emilia_yodas --sources emilia_yodas >> emilia_teacher.log 2>&1 || goto :fail
echo %date% %time% teacher pass done >> emilia_pipeline.log
"%PY%" scripts\02b_second_opinion.py --sources emilia_yodas >> emilia_second.log 2>&1 || goto :fail
echo %date% %time% DONE >> emilia_pipeline.log
exit /b 0
:fail
echo %date% %time% FAILED (exit %errorlevel%) >> emilia_pipeline.log
exit /b 1
