@echo off
echo ==========================================================
echo 🚀 Starting FFmpeg setup on Windows (via CMD)
echo ==========================================================

:: Step 0: Check if FFmpeg is already installed
echo [0/5] Checking existing FFmpeg installation...
ffmpeg -version >nul 2>&1
if %errorlevel%==0 (
    echo ✅ FFmpeg is already installed and accessible!
    echo Skipping download and setup.
    echo ----------------------------------------------------------
    ffmpeg -version | findstr "version"
    echo ----------------------------------------------------------
    pause
    exit /b
) else (
    echo ⚠️  FFmpeg not found. Proceeding with installation...
)

:: Step 1: Create a working directory
echo [1/5] Creating folder C:\ffmpeg ...
mkdir C:\ffmpeg >nul 2>&1
cd /d C:\ffmpeg

:: Step 2: Download latest FFmpeg build from gyan.dev
echo [2/5] Downloading FFmpeg ZIP file...
powershell -Command "Invoke-WebRequest -Uri https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip -OutFile ffmpeg.zip"
if not exist ffmpeg.zip (
    echo ❌ Download failed. Check your internet connection.
    exit /b
)

:: Step 3: Extract the archive
echo [3/5] Extracting FFmpeg...
powershell -Command "Expand-Archive -Path ffmpeg.zip -DestinationPath C:\ffmpeg" >nul 2>&1

:: Find the extracted folder automatically
for /d %%i in (C:\ffmpeg\ffmpeg-*-essentials_build) do set "FFMPEG_DIR=%%i"

if not defined FFMPEG_DIR (
    echo ❌ Could not find extracted folder.
    exit /b
)

:: Step 4: Create safe environment variable
echo [4/5] Setting FFMPEG_PATH variable...
setx FFMPEG_PATH "%FFMPEG_DIR%\bin" >nul
set "PATH=%PATH%;%FFMPEG_DIR%\bin"

:: Step 5: Verify installation
echo [5/5] Verifying FFmpeg installation...
ffmpeg -version >nul 2>&1

if %errorlevel%==0 (
    echo ✅ FFmpeg successfully installed and available!
) else (
    echo ⚠️  FFmpeg not found in PATH yet. Try opening a new CMD and running "ffmpeg -version".
)

:: Cleanup (optional)
del C:\ffmpeg\ffmpeg.zip >nul 2>&1

echo ==========================================================
echo ✅ Setup complete. You can now run FFmpeg commands.
echo ==========================================================
pause
