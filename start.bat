@echo off
chcp 65001 >nul
echo ===================================================
echo   🚀 Lung-Ai Pro 통합 서버 실행 매니저 🚀
echo ===================================================
echo.

:: 1. 파이썬 서버 실행 (가상환경 .venv 자동 활성화 포함)
echo [1] 파이썬 AI 백엔드를 시작합니다...
start "Python FastAPI Server" cmd /k "if exist .venv\Scripts\activate (call .venv\Scripts\activate) else (echo 가상환경이 없습니다.) && uvicorn main:app --reload"

:: 파이썬 서버가 켜질 시간을 2초 정도 벌어줍니다.
timeout /t 2 >nul

:: 2. 리액트 프론트엔드 실행
echo [2] 리액트 웹 프론트엔드를 시작합니다...
cd lung-ai-web
start "React Web Server" cmd /k "npm run dev"

echo.
echo ✅ 모든 서버 실행 준비가 완료되었습니다!
echo ✅ 방금 새로 뜬 두 개의 검은 창(터미널)을 끄지 마세요.
echo ✅ 브라우저에서 http://localhost:5173/ 으로 접속하세요.
echo.
pause