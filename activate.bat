@echo off
REM F1 Teleop Pipeline - 激活开发环境
call C:\ProgramData\miniconda3\Scripts\activate.bat f1-teleop
echo.
echo ============================================
echo   F1 Teleop Pipeline 开发环境已激活
echo   Python: f1-teleop (conda env)
echo   工作目录: %CD%
echo ============================================
echo.
echo 可用脚本:
echo   python scripts\01_test_environment.py   环境验证
echo   python scripts\02_record_episode.py     录制演示
echo   python scripts\03_convert_to_lerobot.py 转LeRobot格式
echo.
cmd /k
