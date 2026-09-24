@echo off
rem Resumable second-opinion pass (02b) for galgame, the one train source that needs kotoba, under the auto-restart
rem supervisor. Safe to re-run. --watch-dir must be the output dir of the single source in --sources: the supervisor's
rem one-shard blocking mode counts the files there, and counting all of second_out (emilia_yodas finished out of
rem manifest order) would send the whole remainder through CUDA_LAUNCH_BLOCKING=1 instead.
cd /d D:\Shizu-ko-distill
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
"C:\Users\multy\AppData\Local\Programs\Python\Python312\python.exe" scripts\run_teacher_pass.py --script scripts\02b_second_opinion.py --watch-dir second_out\galgame --done-suffix .jsonl --sources galgame --batch 24 >> galgame_second.log 2>&1
