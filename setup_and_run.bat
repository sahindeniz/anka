@echo off
setlocal enabledelayedexpansion
title Astro Maestro Pro

:: ── Klasör ───────────────────────────────────────────────────────────────
set "APP_DIR=%~dp0"
set "APP_DIR=%APP_DIR:~0,-1%"
cd /d "%APP_DIR%"

:: ── Python kontrolü ──────────────────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo  Python bulunamadi!
    echo  install.bat dosyasini calistirarak once kurulum yapin.
    echo.
    pause
    exit /b 1
)

:: ── Hızlı paket kontrolü ─────────────────────────────────────────────────
python -c "import PyQt6, numpy, cv2, astropy" >nul 2>&1
if errorlevel 1 (
    echo  Eksik paketler tespit edildi. Yukleniyor...
    python -m pip install numpy opencv-python astropy scikit-image scipy ^
        matplotlib PyQt6 astroquery pyyaml tifffile PyWavelets ^
        --quiet --disable-pip-version-check
)

:: ── Başlat ───────────────────────────────────────────────────────────────
echo  Uygulama baslatiliyor...
echo.
python main.py 2>&1
set "EXIT_CODE=%errorlevel%"
echo.
if not "%EXIT_CODE%"=="0" (
    echo  ============================================================
    echo  HATA: Program beklenmedik sekilde kapandi. (kod: %EXIT_CODE%)
    echo  Yukaridaki hata mesajlarini inceleyin.
    echo  ============================================================
)
echo.
echo  Kapatmak icin bir tusa basin...
pause >nul
