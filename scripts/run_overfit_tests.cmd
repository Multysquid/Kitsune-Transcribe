@echo off
rem Laptop overfit sanity runs of scripts/04_distill.py with the real 0.6B student, one after the other:
rem configs/overfit_1s.json (1000 epochs), overfit_10s.json (100 epochs), overfit_1h.json (30 epochs). Each runs
rem under scripts/supervise_distill.py, which relaunches it with --resume from its newest full state after a crash
rem (this GPU faults now and then under load). A run that still fails does not stop the next one. Console output of
rem all three: overfit_tests.log; start, end and exit code of each run and of each attempt: overfit_pipeline.log.
rem Results: runs\overfit-*\ (TensorBoard: tensorboard --logdir runs). Needs the GPU to itself.
cd /d D:\Shizu-ko-distill
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set PY=C:\Users\multy\AppData\Local\Programs\Python\Python312\python.exe
echo %date% %time% START >> overfit_pipeline.log
call :run overfit_1s
call :run overfit_10s
call :run overfit_1h
echo %date% %time% DONE >> overfit_pipeline.log
exit /b 0

:run
echo %date% %time% start %1 >> overfit_pipeline.log
"%PY%" scripts\supervise_distill.py --config configs\%1.json --log overfit_pipeline.log >> overfit_tests.log 2>&1
echo %date% %time% end %1 (exit %errorlevel%) >> overfit_pipeline.log
exit /b 0
