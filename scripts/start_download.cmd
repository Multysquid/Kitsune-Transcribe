@echo off
rem Resumable full-dataset download. Safe to re-run: continues from the last finished input file.
rem Runs the checkout this file is in (the parent of scripts\), not a fixed path; PY overrides the interpreter.
rem It downloads into that checkout's data\ (a git worktree has none: it would start from scratch).
cd /d "%~dp0.." || exit /b 1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
if not defined PY set "PY=C:\Users\multy\AppData\Local\Programs\Python\Python312\python.exe"
"%PY%" scripts\01_prepare_data.py --sources galgame reazon_medium --galgame-shards 115 >> prepare_full.log 2>&1
