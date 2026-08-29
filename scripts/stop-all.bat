@echo off
REM ============================================================
REM CipherMind Sentinel - stop the local stack (Windows)
REM ============================================================
echo stopping soc-engine ...
taskkill /FI "WINDOWTITLE eq soc-engine*" /F >nul 2>&1
taskkill /F /IM bun.exe /FI "WINDOWTITLE eq soc-engine*" >nul 2>&1
echo stopping Next.js dev server ...
taskkill /FI "WINDOWTITLE eq ciphermind-frontend*" /F >nul 2>&1
wmic process where "commandline like '%%next dev%%'" call terminate >nul 2>&1
echo done.
