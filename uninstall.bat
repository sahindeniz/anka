@echo off
setlocal enabledelayedexpansion
title Astromastro - Kaldirma

echo.
echo  ================================================================
echo    ASTROMASTRO - Program Kaldirma
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
if exist "%USERPROFILE%\Desktop\Astromastro.lnk" (
    del "%USERPROFILE%\Desktop\Astromastro.lnk"
    echo  [OK] Masaustu kisayolu silindi.
)
if exist "%USERPROFILE%\Desktop\AstroMestro.lnk" (
    del "%USERPROFILE%\Desktop\AstroMestro.lnk"
)
if exist "%USERPROFILE%\Desktop\Astro Maestro Pro.lnk" (
    del "%USERPROFILE%\Desktop\Astro Maestro Pro.lnk"
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
