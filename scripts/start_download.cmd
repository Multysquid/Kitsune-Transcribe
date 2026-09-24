@echo off
rem Resumable full-dataset download. Safe to re-run: continues from the last finished input file.
cd /d D:\Shizu-ko-distill
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
"C:\Users\multy\AppData\Local\Programs\Python\Python312\python.exe" scripts\01_prepare_data.py --sources galgame reazon_medium --galgame-shards 115 >> prepare_full.log 2>&1
