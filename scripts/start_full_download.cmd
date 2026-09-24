@echo off
rem Full-scale datasets for the real run (~440 GB on top of what is already in data\), smallest first:
rem   galgame (all 115 tars) -> emilia_yodas (every tar but the eval_emilia hold-out's) -> emilia_nc (CC BY-NC)
rem   -> reazon_large (ReazonSpeech ~5000 h; rows already in reazon_small are skipped).
rem Resumable: re-run after a crash, reboot or a full disk (01_prepare_data stops before any download that would
rem leave less than 30 GB free). Each source is attempted even if an earlier one failed.
cd /d D:\Shizu-ko-distill
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set PY=C:\Users\multy\AppData\Local\Programs\Python\Python312\python.exe
echo %date% %time% START >> full_download_pipeline.log
call :run galgame --galgame-shards 115
call :run emilia_yodas --emilia-hours 100000
call :run emilia_nc
call :run reazon_large
echo %date% %time% DONE >> full_download_pipeline.log
exit /b 0

:run
echo %date% %time% start %1 >> full_download_pipeline.log
"%PY%" scripts\01_prepare_data.py --sources %* >> full_download.log 2>&1
echo %date% %time% end %1 (exit %errorlevel%) >> full_download_pipeline.log
exit /b 0
