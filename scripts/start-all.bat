@echo off
REM ============================================================
REM CipherMind Sentinel - start the full stack locally (Windows)
REM   1. soc-engine  : ML inference engine  -^> http://localhost:3010
REM   2. frontend    : Next.js SOC UI       -^> http://localhost:3000
REM
REM Requirements: Bun (https://bun.sh) on PATH.
REM First run installs dependencies. Stop with: scripts/stop-all.bat
REM ============================================================
setlocal
cd /d "%~dp0.."

where bun >nul 2>&1
if errorlevel 1 (
  echo ERROR: bun is not installed - see https://bun.sh
  exit /b 1
)

if not exist .env (
  copy .env.example .env >nul
  echo [env] created .env from .env.example
)

echo == CipherMind Sentinel - starting ==

REM ------------------------------------------------ soc-engine :3010
curl -s -m 3 http://localhost:3010/api/health >nul 2>&1
if errorlevel 1 (
  if not exist mini-services\soc-engine\node_modules (
    echo [engine]   installing dependencies ^(first run^) ...
    pushd mini-services\soc-engine
    call bun install
    popd
  )
  echo [engine]   starting soc-engine on :3010 ...
  pushd mini-services\soc-engine
  start "soc-engine" cmd /c "bun run dev > engine.log 2>&1"
  popd
) else (
  echo [engine]   already running on :3010
)

REM ------------------------------------------------ frontend :3000
curl -s -m 3 -o nul http://localhost:3000/ 2>nul
if errorlevel 1 (
  if not exist node_modules (
    echo [frontend] installing dependencies ^(first run - can take a few minutes^) ...
    call bun install
  )
  echo [frontend] starting Next.js dev server on :3000 ...
  start "ciphermind-frontend" cmd /c "bun run dev > frontend.log 2>&1"
) else (
  echo [frontend] already running on :3000
)

echo.
echo [wait]     engine boot takes ~20-60s ^(scores 12,000 boot events^)
echo ============================================================
echo   Frontend ^(SOC UI^) : http://localhost:3000
echo   Engine health     : http://localhost:3010/api/health
echo   Stop the stack    : scripts\stop-all.bat
echo ============================================================
endlocal
