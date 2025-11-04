@echo off
REM Windows Setup Script for Video Summarizer
REM Optimized for Dual NVIDIA Quadro RTX 8000

echo ================================================
echo Video Summarizer - Windows Setup
echo Dual NVIDIA Quadro RTX 8000 Detected
echo ================================================
echo.

REM Check Python installation
echo [1/7] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found! Please install Python 3.8+
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)
python --version
echo.

REM Check NVIDIA GPU
echo [2/7] Checking NVIDIA GPUs...
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo ERROR: nvidia-smi not found! Please install NVIDIA drivers
    pause
    exit /b 1
)
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo.

REM Create virtual environment
echo [3/7] Creating virtual environment...
if exist venv (
    echo Virtual environment already exists, skipping...
) else (
    python -m venv venv
    echo Virtual environment created!
)
echo.

REM Activate virtual environment
echo [4/7] Activating virtual environment...
call venv\Scripts\activate.bat
echo.

REM Upgrade pip
echo [5/7] Upgrading pip...
python -m pip install --upgrade pip
echo.

REM Install PyTorch with CUDA
echo [6/7] Installing PyTorch with CUDA support...
echo This may take a few minutes...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
echo.

REM Install other requirements
echo [7/7] Installing other dependencies...
pip install -r requirements.txt
echo.

REM Create directory structure
echo Creating directory structure...
if not exist data\train\videos mkdir data\train\videos
if not exist data\val\videos mkdir data\val\videos
if not exist data\test\videos mkdir data\test\videos
if not exist outputs mkdir outputs
if not exist checkpoints mkdir checkpoints
if not exist cache mkdir cache
echo Directory structure created!
echo.

REM Copy optimized config
echo Setting up optimized configuration...
if exist config_rtx8000_optimized.yaml (
    copy config_rtx8000_optimized.yaml config.yaml
    echo Optimized config copied to config.yaml
) else (
    echo Warning: config_rtx8000_optimized.yaml not found
)
echo.

REM Test GPU setup
echo Testing GPU configuration...
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Count: {torch.cuda.device_count()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"
echo.

echo ================================================
echo Setup Complete! ✓
echo ================================================
echo.
echo Next steps:
echo 1. Add your videos to data\train\videos\
echo 2. Create data\train\summaries.json
echo 3. Run: python preprocess.py --split train --video_dir data\train\videos --summaries_file data\train\summaries.json
echo 4. Run: python train.py --config config.yaml
echo.
echo For more info, see README.md and QUICKSTART.md
echo.
pause