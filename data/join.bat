@echo off

cd ..\tools
python join_data.py --join_dir_prefix=..\data\data --out_dir=..\data\out
pause
