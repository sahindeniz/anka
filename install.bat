@echo off
setlocal enabledelayedexpansion
title Astro Maestro Pro — Kurulum

color 0B
echo.
echo  ================================================================
echo    ASTRO MAESTRO PRO — Kurulum Sihirbazi
echo    by Deniz
echo  ================================================================
echo.

:: ── Yönetici kontrolü ───────────────────────────────────────────────────
net session >nul 2>&1
if errorlevel 1 (
    echo  [!] Bu betik yonetici yetkisi gerektirmez.
    echo.
)

:: ── Kurulum klasörü ──────────────────────────────────────────────────────
set "INSTALL_DIR=%~dp0"
set "INSTALL_DIR=%INSTALL_DIR:~0,-1%"
echo  [i] Kurulum klasoru: %INSTALL_DIR%
echo.

:: ── Python kontrolü ──────────────────────────────────────────────────────
echo  [1/5] Python kontrolu...
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo  [!] Python bulunamadi!
    echo.
    echo  Python 3.10 veya 3.11 kurulu olmalidir:
    echo    https://www.python.org/downloads/
    echo.
    echo  Kurulum sirasinda "Add Python to PATH" secenegini isaretleyin!
    echo.
    echo  Python kurduktan sonra bu betigi tekrar calistirin.
    echo.
    start https://www.python.org/downloads/
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('python --version') do set PYVER=%%i
echo  [OK] %PYVER% bulundu.

:: Python versiyon kontrolü (3.9 minimum)
python -c "import sys; exit(0 if sys.version_info >= (3,9) else 1)" >nul 2>&1
if errorlevel 1 (
    echo  [!] Python 3.9 veya uzeri gereklidir. Lutfen guncelleyin.
    pause
    exit /b 1
)

:: ── pip kontrolü ─────────────────────────────────────────────────────────
echo.
echo  [2/5] pip kontrolu...
python -m pip --version >nul 2>&1
if errorlevel 1 (
    echo  [!] pip bulunamadi. Yukleniyor...
    python -m ensurepip --upgrade
)
python -m pip install --upgrade pip --quiet --disable-pip-version-check
echo  [OK] pip hazir.

:: ── Gerekli paketler ─────────────────────────────────────────────────────
echo.
echo  [3/5] Gerekli Python paketleri yukleniyor...
echo        (Ilk kurulumda 3-8 dakika surebilir)
echo.

set PACKAGES=numpy opencv-python astropy scikit-image scipy matplotlib PyQt6 astroquery pyyaml tifffile PyWavelets

for %%p in (%PACKAGES%) do (
    echo    Yukleniyor: %%p ...
    python -m pip install %%p --quiet --disable-pip-version-check
    if errorlevel 1 (
        echo    [!] %%p yuklenemedi, tekrar deneniyor...
        python -m pip install %%p --quiet --disable-pip-version-check --no-cache-dir
    )
)

echo.
echo  [OK] Tum paketler yuklendi.

:: ── Yapılandırma ──────────────────────────────────────────────────────────
echo.
echo  [4/5] Yapilandiriliyor...

:: settings.json yoksa oluştur
if not exist "%INSTALL_DIR%\settings.json" (
    python -c "import json; json.dump({}, open('%INSTALL_DIR%/settings.json','w'))"
    echo  [OK] settings.json olusturuldu.
) else (
    echo  [OK] Mevcut ayarlar korundu.
)

:: ── Kısayol oluşturma ─────────────────────────────────────────────────────
echo.
echo  [5/5] Masaustu kisayolu olusturuluyor...

set "SHORTCUT_PATH=%USERPROFILE%\Desktop\Astro Maestro Pro.lnk"
set "LAUNCHER=%INSTALL_DIR%\setup_and_run.bat"

:: PowerShell ile kısayol oluştur
powershell -Command ^
  "$ws = New-Object -ComObject WScript.Shell; " ^
  "$sc = $ws.CreateShortcut('%SHORTCUT_PATH%'); " ^
  "$sc.TargetPath = '%LAUNCHER%'; " ^
  "$sc.WorkingDirectory = '%INSTALL_DIR%'; " ^
  "$sc.Description = 'Astro Maestro Pro'; " ^
  "$sc.Save()" >nul 2>&1

if exist "%SHORTCUT_PATH%" (
    echo  [OK] Masaustu kisayolu olusturuldu.
) else (
    echo  [!] Kisayol olusturulamadi (normal, manuel olusturabilirsiniz).
)

:: ── Tamamlandı ────────────────────────────────────────────────────────────
echo.
echo  ================================================================
echo    KURULUM TAMAMLANDI!
echo  ================================================================
echo.
echo    Programi baslatmak icin:
echo      • Masaustundeki "Astro Maestro Pro" kisayoluna cift tiklayin
echo      • Ya da: setup_and_run.bat dosyasina cift tiklayin
echo.
echo    Ilk acilista gecmis ayarlar yoksa program otomatik baslayacak.
echo.

set /p LAUNCH="  Programi simdi baslatmak istiyor musunuz? (E/H): "
if /i "!LAUNCH!" == "E" (
    echo.
    echo  Baslatiliyor...
    start "" /B python "%INSTALL_DIR%\main.py"
)

echo.
pause
