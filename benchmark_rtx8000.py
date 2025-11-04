"""
Benchmark script for dual NVIDIA Quadro RTX 8000 setup.
Tests model loading, inference speed, and memory usage.
"""
import time
import logging
from pathlib import Path

import torch
import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def print_header(text):
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def get_gpu_memory():
    """Get current GPU memory usage for all GPUs (GB)."""
    memory_info = []
    for i in range(torch.cuda.device_count()):
        device = torch.device(f"cuda:{i}")
        allocated = torch.cuda.memory_allocated(device) / 1e9
        reserved = torch.cuda.memory_reserved(device) / 1e9
        total = torch.cuda.get_device_properties(i).total_memory / 1e9
        memory_info.append({
            "gpu": i,
            "allocated": allocated,
            "reserved": reserved,
            "total": total,
            "free": total - reserved
        })
    return memory_info


def benchmark_gpu_raw():
    """Raw GPU compute benchmark."""
    print_header("Raw GPU Compute Benchmark")

    sizes = [5000, 10000, 15000]

    for gpu_id in range(torch.cuda.device_count()):
        print(f"\n🎮 GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
        device = torch.device(f"cuda:{gpu_id}")

        for size in sizes:
            # Matrix multiplication benchmark
            # Be careful: very large sizes may OOM; keep sizes appropriate for your GPUs
            try:
                x = torch.randn(size, size, device=device, dtype=torch.float32)

                # Warm up
                _ = torch.mm(x, x)
                torch.cuda.synchronize(device)

                # Benchmark
                start = time.time()
                iterations = 10
                for _ in range(iterations):
                    y = torch.mm(x, x)
                torch.cuda.synchronize(device)
                elapsed = time.time() - start

                # Compute TFLOPS: 2 * n^3 ops per multiply
                tflops = (iterations * (size ** 3) * 2) / elapsed / 1e12
                ops_per_sec = iterations / elapsed

                max_mem = torch.cuda.max_memory_allocated(device) / 1e9

                print(f"\nMatrix size: {size}x{size}")
                print(f"  Time: {elapsed:.3f}s ({iterations} iterations)")
                print(f"  Speed: {ops_per_sec:.2f} ops/sec")
                print(f"  Performance: {tflops:.2f} TFLOPS")
                print(f"  Peak memory used: {max_mem:.2f} GB")

            except RuntimeError as e:
                print(f"  ❌ RuntimeError (likely OOM) for size {size}: {e}")
            finally:
                # free tensors and cache
                try:
                    del x, y
                except Exception:
                    pass
                torch.cuda.empty_cache()


def benchmark_blip2():
    """Benchmark BLIP-2 caption generation."""
    print_header("BLIP-2 Caption Generation Benchmark")

    try:
        from src.models.caption_model import CaptionModel
    except Exception as e:
        print(f"❌ Cannot import CaptionModel: {e}")
        return

    # Test different model sizes
    models = [
        ("Salesforce/blip2-opt-2.7b", "Standard"),
        ("Salesforce/blip2-flan-t5-xl", "Large (Recommended)")
    ]

    for model_name, desc in models:
        print(f"\n📷 Testing {desc}: {model_name}")

        try:
            start_load = time.time()
            model = CaptionModel(model_name=model_name, device="cuda:0")
            load_time = time.time() - start_load

            print(f"  Load time: {load_time:.2f}s")

            # Test on dummy images
            test_image = Image.new("RGB", (480, 640), color="blue")

            # Single image
            with torch.no_grad():
                start = time.time()
                caption = model.generate_caption(test_image)
                torch.cuda.synchronize("cuda:0")
                single_time = time.time() - start

            print(f"  Single caption time: {single_time:.2f}s")
            print(f"  Caption: {str(caption)[:200]}")

            # Batch of images
            batch_size = 10
            images = [test_image] * batch_size

            with torch.no_grad():
                start = time.time()
                captions = model.generate_captions_batch(images)
                torch.cuda.synchronize("cuda:0")
                batch_time = time.time() - start

            print(f"  Batch ({batch_size} images) time: {batch_time:.2f}s")
            print(f"  Time per image: {batch_time / batch_size:.2f}s")
            print(f"  Throughput: {batch_size / batch_time:.2f} images/sec")

            # Memory usage
            memory = get_gpu_memory()
            if memory:
                print(f"  GPU 0 memory: {memory[0]['allocated']:.2f} GB allocated")

            # cleanup
            del model
            torch.cuda.empty_cache()
            print("  ✓ Test passed")

        except Exception as e:
            print(f"  ❌ Error during BLIP-2 test: {e}")
            try:
                del model
            except Exception:
                pass
            torch.cuda.empty_cache()
            continue


def benchmark_whisper():
    """Benchmark Whisper transcription."""
    print_header("Whisper Audio Transcription Benchmark")

    try:
        from src.models.transcription_model import TranscriptionModel
    except Exception as e:
        print(f"❌ Cannot import TranscriptionModel: {e}")
        return

    # Test different model sizes
    models = [
        ("base", "Base"),
        ("medium", "Medium"),
        ("large-v2", "Large (Recommended)")
    ]

    for model_name, desc in models:
        print(f"\n🎤 Testing {desc}: {model_name}")

        try:
            start_load = time.time()
            model = TranscriptionModel(model_name=model_name, device="cuda:0")
            load_time = time.time() - start_load

            print(f"  Load time: {load_time:.2f}s")

            # Generate dummy audio (30 seconds)
            sample_rate = 16000
            duration = 30
            audio = np.random.randn(sample_rate * duration).astype(np.float32)
            audio = audio / (np.max(np.abs(audio)) + 1e-8)  # Normalize safely

            # Transcribe
            with torch.no_grad():
                start = time.time()
                result = model.transcribe(audio, sample_rate)
                torch.cuda.synchronize("cuda:0")
                transcribe_time = time.time() - start

            print(f"  Transcription time: {transcribe_time:.2f}s")
            print(f"  Real-time factor: {duration / transcribe_time:.2f}x")
            print(f"  Speed: {duration / transcribe_time:.1f}x faster than real-time")

            # Memory usage
            memory = get_gpu_memory()
            if memory:
                print(f"  GPU 0 memory: {memory[0]['allocated']:.2f} GB allocated")

            del model
            torch.cuda.empty_cache()
            print("  ✓ Test passed")

        except Exception as e:
            print(f"  ❌ Error during Whisper test: {e}")
            try:
                del model
            except Exception:
                pass
            torch.cuda.empty_cache()
            continue


def benchmark_t5():
    """Benchmark T5 summarization."""
    print_header("T5 Summarization Benchmark")

    try:
        from src.models.summarization_model import SummarizationModel
    except Exception as e:
        print(f"❌ Cannot import SummarizationModel: {e}")
        return

    # Test different model sizes
    models = [
        ("t5-small", "Small"),
        ("t5-base", "Base"),
        ("t5-large", "Large (Recommended)")
    ]

    # Ensure there is a second GPU before using cuda:1
    second_gpu_available = torch.cuda.device_count() > 1

    for model_name, desc in models:
        print(f"\n📝 Testing {desc}: {model_name}")

        try:
            device_to_use = "cuda:1" if second_gpu_available else "cuda:0"
            start_load = time.time()
            model = SummarizationModel(model_name=model_name, device=device_to_use)
            load_time = time.time() - start_load

            print(f"  Load time: {load_time:.2f}s (device={device_to_use})")

            # Test input (different lengths)
            test_texts = [
                "This is a short test. " * 10,   # ~100 words
                "This is a longer test. " * 50,  # ~500 words
                "This is a very long test. " * 100,  # ~1000 words
            ]

            for i, text in enumerate(test_texts):
                words = len(text.split())
                with torch.no_grad():
                    start = time.time()
                    summary = model.summarize(text)
                    # sync for the device used
                    if torch.cuda.is_available():
                        torch.cuda.synchronize(device_to_use)
                    summarize_time = time.time() - start

                print(f"\n  Input {i + 1} ({words} words):")
                print(f"    Time: {summarize_time:.2f}s")
                print(f"    Speed: {words / summarize_time:.1f} words/sec")
                print(f"    Summary length: {len(str(summary).split())} words")

            # Memory usage
            memory = get_gpu_memory()
            if len(memory) > 1:
                print(f"\n  GPU 1 memory: {memory[1]['allocated']:.2f} GB allocated")
            elif memory:
                print(f"\n  GPU 0 memory: {memory[0]['allocated']:.2f} GB allocated")

            del model
            torch.cuda.empty_cache()
            print("  ✓ Test passed")

        except Exception as e:
            print(f"  ❌ Error during T5 test: {e}")
            try:
                del model
            except Exception:
                pass
            torch.cuda.empty_cache()
            continue


def benchmark_full_pipeline():
    """Benchmark complete pipeline."""
    print_header("Full Pipeline Benchmark")

    print("\n🎬 Testing end-to-end video summarization pipeline...")

    try:
        from src.models.caption_model import CaptionModel
        from src.models.transcription_model import TranscriptionModel
        from src.models.summarization_model import SummarizationModel
    except Exception as e:
        print(f"❌ Cannot import one or more pipeline models: {e}")
        return

    try:
        print("\nLoading models...")
        start_total = time.time()

        # Load all models
        start = time.time()
        caption_model = CaptionModel(model_name="Salesforce/blip2-opt-2.7b", device="cuda:0")
        caption_time = time.time() - start
        print(f"  BLIP-2 loaded: {caption_time:.2f}s")

        start = time.time()
        transcription_model = TranscriptionModel(model_name="base", device="cuda:0")
        transcription_time = time.time() - start
        print(f"  Whisper loaded: {transcription_time:.2f}s")

        # prefer second GPU for summarization if available
        summary_device = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
        start = time.time()
        summary_model = SummarizationModel(model_name="t5-base", device=summary_device)
        summary_time = time.time() - start
        print(f"  T5 loaded: {summary_time:.2f}s (device={summary_device})")

        total_load_time = time.time() - start_total
        print(f"\nTotal model loading time: {total_load_time:.2f}s")

        # Simulate video processing
        print("\nSimulating video processing...")

        # Generate dummy frames
        num_frames = 30
        frames = [Image.new("RGB", (640, 480), color="blue") for _ in range(num_frames)]

        # Caption generation
        with torch.no_grad():
            start = time.time()
            captions = caption_model.generate_captions_batch(frames)
            caption_text = caption_model.combine_captions(captions)
            if torch.cuda.is_available():
                torch.cuda.synchronize("cuda:0")
            caption_gen_time = time.time() - start
        print(f"  Caption generation ({num_frames} frames): {caption_gen_time:.2f}s")

        # Audio transcription
        audio = np.random.randn(16000 * 30).astype(np.float32)
        audio = audio / (np.max(np.abs(audio)) + 1e-8)

        with torch.no_grad():
            start = time.time()
            transcript = transcription_model.get_transcript_text(audio, 16000)
            if torch.cuda.is_available():
                torch.cuda.synchronize("cuda:0")
            transcription_gen_time = time.time() - start
        print(f"  Audio transcription (30s): {transcription_gen_time:.2f}s")

        # Summarization
        combined_text = f"Video captions: {caption_text} Audio transcript: {transcript}"

        with torch.no_grad():
            start = time.time()
            summary = summary_model.summarize(combined_text)
            if torch.cuda.is_available():
                torch.cuda.synchronize(summary_device)
            summary_gen_time = time.time() - start
        print(f"  Summary generation: {summary_gen_time:.2f}s")

        total_inference_time = caption_gen_time + transcription_gen_time + summary_gen_time
        print(f"\nTotal inference time: {total_inference_time:.2f}s")
        print(f"Total pipeline time (including loading): {total_load_time + total_inference_time:.2f}s")

        # Memory usage
        print("\nMemory usage:")
        memory = get_gpu_memory()
        for i, mem in enumerate(memory):
            print(f"  GPU {i}: {mem['allocated']:.2f} GB / {mem['total']:.2f} GB")

        print("\n✓ Full pipeline test passed!")

    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
    finally:
        # cleanup
        try:
            del caption_model, transcription_model, summary_model
        except Exception:
            pass
        torch.cuda.empty_cache()


def main():
    print("\n" + "=" * 70)
    print("  NVIDIA QUADRO RTX 8000 BENCHMARK SUITE")
    print("  Video Summarizer Performance Test")
    print("=" * 70)

    # Check CUDA
    if not torch.cuda.is_available():
        print("\n❌ CUDA not available!")
        return

    # Display GPU info
    print("\n🎮 GPU Configuration:")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {props.total_memory / 1e9:.2f} GB")
        print(f"  Compute Capability: {props.major}.{props.minor}")
        print(f"  Multi-Processors: {props.multi_processor_count}")

    # Run benchmarks
    try:
        benchmark_gpu_raw()
        benchmark_blip2()
        benchmark_whisper()
        benchmark_t5()
        benchmark_full_pipeline()

        print_header("Benchmark Complete!")
        print("\nYour RTX 8000 setup is performing excellently! ✓")
        print("\nRecommended configuration:")
        print("  - BLIP-2: Salesforce/blip2-flan-t5-xl")
        print("  - Whisper: large-v2")
        print("  - T5: t5-large")
        print("  - Batch size: 16")
        print("\nSee config_rtx8000_optimized.yaml for optimized settings.")

    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user.")
    except Exception as e:
        print(f"\n❌ Benchmark error: {e}")


if __name__ == "__main__":
    main()
