@echo off
chcp 65001 >nul
echo YuE 서버를 종료합니다...
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | Where-Object { .CommandLine -like '*gradio_server.py*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"
echo 종료 완료.
timeout /t 2 >nul
