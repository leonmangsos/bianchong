@echo off
title 编钟体感演奏器
echo 正在启动编钟程序...
cd /d "%~dp0"
if exist bianchong.exe (
    start bianchong.exe
) else (
    echo [ERROR] bianchong.exe not found!
    echo Attempting to run Python script...
    python bianchong.py
)
exit