@echo off
rem Resumable second-opinion pass (02b) under the auto-restart supervisor. Safe to re-run.
cd /d D:\Shizu-ko-distill
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
"C:\Users\multy\AppData\Local\Programs\Python\Python312\python.exe" scripts\run_teacher_pass.py --script scripts\02b_second_opinion.py --watch-dir second_out --done-suffix .jsonl --batch 24 >> second_opinion.log 2>&1
