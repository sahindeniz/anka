@echo off
setlocal enabledelayedexpansion
title Astro Maestro Pro — Kaldirma

echo.
echo  ================================================================
echo    ASTRO MAESTRO PRO — Program Kaldirma
echo  ================================================================
echo.
echo  Bu islem sadece masaustu kisayolunu ve ayarlari kaldirir.
echo  Program dosyalari el ile silinmelidir.
echo.

set /p CONFIRM="  Devam etmek istiyor musunuz? (E/H): "
if /i not "!CONFIRM!" == "E" (
    echo  Iptal edildi.
    pause
    exit /b 0
)

:: Masaüstü kısayolu
if exist "%USERPROFILE%\Desktop\Astro Maestro Pro.lnk" (
    del "%USERPROFILE%\Desktop\Astro Maestro Pro.lnk"
    echo  [OK] Masaustu kisayolu silindi.
)

:: Ayarlar (opsiyonel)
set /p DELSETTINGS="  Ayarlari (settings.json) da silmek istiyor musunuz? (E/H): "
if /i "!DELSETTINGS!" == "E" (
    if exist "%~dp0settings.json" (
        del "%~dp0settings.json"
        echo  [OK] settings.json silindi.
    )
    if exist "%~dp0user_scripts.json" (
        del "%~dp0user_scripts.json"
        echo  [OK] user_scripts.json silindi.
    )
)

echo.
echo  Kaldirma tamamlandi.
echo  Program klasorunu el ile silebilirsiniz: %~dp0
echo.
pause
