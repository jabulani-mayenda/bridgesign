@echo off
title BridgeSign — Collection Session
color 0A

echo.
echo ============================================================
echo   BridgeSign — Full Session Collector
echo   200 samples per sign  ^|  26 signs total
echo   Different location/lighting each session = better model!
echo ============================================================
echo.
echo   You will be guided through all 26 letters one by one.
echo   For each letter:
echo     [S] = Start recording
echo     [Q] = Move on to the next letter
echo.
echo   Press any key to begin the session...
pause >nul

set SIGNS=A B C D E F G H I J K L M N O P Q R S T U V W X Y Z
set COUNT=0
set TOTAL=26

for %%S in (%SIGNS%) do (
    set /a COUNT+=1
    echo.
    echo ------------------------------------------------------------
    echo   [!COUNT! / %TOTAL%]  Collecting: %%S
    echo ------------------------------------------------------------
    python data_collector.py --sign %%S --samples 200
    if errorlevel 1 (
        echo   [!] Something went wrong with %%S. Skipping...
    )
)

echo.
echo ============================================================
echo   Session Complete! All 26 signs collected.
echo   Great work — run this again from a different spot/light!
echo ============================================================
echo.
pause
