"""
Installation test script for Video Summarizer.
Verifies all dependencies and GPU configuration.
"""
import sys
import subprocess
from pathlib import Path

def print_section(title):
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def check_python():
    """Check Python version."""
    print_section("Python Version")
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ ERROR: Python 3.8+ required")
        return False
    print("✓ Python version OK")
    return True

def check_cuda():
    """Check CUDA and GPUs."""
    print_section("CUDA & GPU Configuration")
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if not torch.cuda.is_available():
            print("❌ ERROR: CUDA not available")
            return False
        
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        
        num_gpus = torch.cuda.device_count()
        print(f"\nNumber of GPUs: {num_gpus}")
        
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            print(f"\nGPU {i}:")
            print(f"  Name: {torch.cuda.get_device_name(i)}")
            print(f"  Total Memory: {props.total_memory / 1e9:.2f} GB")
            print(f"  Compute Capability: {props.major}.{props.minor}")
            print(f"  Multi-Processor Count: {props.multi_processor_count}")
        
        print("\n✓ CUDA configuration OK")
        return True
        
    except ImportError:
        print("❌ ERROR: PyTorch not installed")
        return False
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return False

def check_dependencies():
    """Check required Python packages."""
    print_section("Python Dependencies")
    
    required_packages = {
        'torch': 'PyTorch',
        'transformers': 'Hugging Face Transformers',
        'whisper': 'OpenAI Whisper',
        'cv2': 'OpenCV',
        'PIL': 'Pillow',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'yaml': 'PyYAML',
        'tqdm': 'tqdm',
        'rouge_score': 'ROUGE Score',
        'nltk': 'NLTK',
    }
    
    missing = []
    for package, name in required_packages.items():
        try:
            if package == 'cv2':
                import cv2
            elif package == 'PIL':
                from PIL import Image
            elif package == 'yaml':
                import yaml
            else:
                __import__(package)
            print(f"✓ {name}")
        except ImportError:
            print(f"❌ {name}")
            missing.append(name)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        return False
    
    print("\n✓ All dependencies installed")
    return True

def check_ffmpeg():
    """Check FFmpeg installation."""
    print_section("FFmpeg")
    
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            print(version_line)
            print("✓ FFmpeg installed")
            return True
        else:
            print("❌ FFmpeg not working properly")
            return False
            
    except FileNotFoundError:
        print("❌ FFmpeg not found in PATH")
        print("\nInstallation instructions:")
        print("  Windows: Download from https://ffmpeg.org/download.html")
        print("  Ubuntu: sudo apt-get install ffmpeg")
        print("  macOS: brew install ffmpeg")
        return False
    except Exception as e:
        print(f"❌ Error checking FFmpeg: {str(e)}")
        return False

def check_project_structure():
    """Check project directory structure."""
    print_section("Project Structure")
    
    required_dirs = [
        'src',
        'src/models',
        'src/data',
        'src/utils',
        'data',
        'data/train',
        'data/val',
        'data/test',
    ]
    
    required_files = [
        'config.yaml',
        'requirements.txt',
        'preprocess.py',
        'train.py',
        'test.py',
        'inference.py',
        'src/models/caption_model.py',
        'src/models/transcription_model.py',
        'src/models/summarization_model.py',
        'src/data/dataset.py',
        'src/utils/video_processor.py',
        'src/utils/metrics.py',
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing_dirs.append(dir_path)
            print(f"❌ Directory missing: {dir_path}")
        else:
            print(f"✓ {dir_path}/")
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
            print(f"❌ File missing: {file_path}")
        else:
            print(f"✓ {file_path}")
    
    if missing_dirs or missing_files:
        print("\n❌ Project structure incomplete")
        return False
    
    print("\n✓ Project structure OK")
    return True

def test_gpu_performance():
    """Quick GPU performance test."""
    print_section("GPU Performance Test")
    
    try:
        import torch
        import time
        
        for i in range(torch.cuda.device_count()):
            device = f"cuda:{i}"
            print(f"\nTesting GPU {i} ({torch.cuda.get_device_name(i)})...")
            
            # Warm up
            x = torch.randn(5000, 5000, device=device)
            y = torch.mm(x, x)
            torch.cuda.synchronize()
            
            # Benchmark
            start = time.time()
            for _ in range(5):
                y = torch.mm(x, x)
            torch.cuda.synchronize()
            elapsed = time.time() - start
            
            tflops = (5 * 5000**3 * 2) / elapsed / 1e12
            print(f"  Time: {elapsed:.3f}s")
            print(f"  Performance: {tflops:.2f} TFLOPS")
            print(f"  Memory allocated: {torch.cuda.memory_allocated(i) / 1e9:.2f} GB")
            print(f"  Memory reserved: {torch.cuda.memory_reserved(i) / 1e9:.2f} GB")
        
        print("\n✓ GPU performance test passed")
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {str(e)}")
        return False

def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("  VIDEO SUMMARIZER - INSTALLATION TEST")
    print("="*60)
    
    results = {
        'Python': check_python(),
        'CUDA & GPU': check_cuda(),
        'Dependencies': check_dependencies(),
        'FFmpeg': check_ffmpeg(),
        'Project Structure': check_project_structure(),
        'GPU Performance': test_gpu_performance(),
    }
    
    print_section("Summary")
    
    all_passed = True
    for component, passed in results.items():
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"{status} - {component}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("  ✓ ALL TESTS PASSED!")
        print("="*60)
        print("\nYour system is ready for video summarization!")
        print("\nNext steps:")
        print("  1. Add videos to data/train/videos/")
        print("  2. Create data/train/summaries.json")
        print("  3. Run: python preprocess.py --split train \\")
        print("            --video_dir data/train/videos \\")
        print("            --summaries_file data/train/summaries.json")
        print("  4. Run: python train.py")
    else:
        print("  ❌ SOME TESTS FAILED")
        print("="*60)
        print("\nPlease fix the issues above before proceeding.")
        print("See README.md for installation instructions.")
    
    print("\n")
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())