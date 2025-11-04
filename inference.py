"""
Inference script for video summarization.
"""
import yaml
import argparse
import logging
from pathlib import Path

from src.utils.video_processor import VideoProcessor
from src.models.caption_model import CaptionModel
from src.models.transcription_model import TranscriptionModel
from src.models.summarization_model import SummarizationModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoSummarizer:
    """End-to-end video summarization pipeline."""
    
    def __init__(self, config: dict, checkpoint_path: str = None):
        """
        Initialize video summarizer.
        
        Args:
            config: Configuration dictionary
            checkpoint_path: Path to fine-tuned T5 checkpoint (optional)
        """
        self.config = config
        
        logger.info("Initializing video summarization pipeline...")
        
        # Initialize video processor
        self.video_processor = VideoProcessor(
            fps=config["video_processing"]["frame_extraction"]["fps"],
            max_frames=config["video_processing"]["frame_extraction"]["max_frames"],
            sample_rate=config["video_processing"]["audio_extraction"]["sample_rate"]
        )
        
        # Initialize caption model
        self.caption_model = CaptionModel(
            model_name=config["models"]["blip2"]["model_name"],
            device=config["models"]["blip2"]["device"]
        )
        
        # Initialize transcription model
        self.transcription_model = TranscriptionModel(
            model_name=config["models"]["whisper"]["model_name"],
            device=config["models"]["whisper"]["device"]
        )
        
        # Initialize summarization model
        model_name = checkpoint_path if checkpoint_path else config["models"]["t5"]["model_name"]
        self.summarization_model = SummarizationModel(
            model_name=model_name,
            device=config["models"]["t5"].get("device", "cuda"),
            max_input_length=config["models"]["t5"]["max_input_length"],
            max_output_length=config["models"]["t5"]["max_output_length"]
        )
        
        logger.info("Pipeline initialized successfully!")
    
    def summarize_video(self, video_path: str, verbose: bool = True) -> dict:
        """
        Generate summary for a video.
        
        Args:
            video_path: Path to video file
            verbose: Whether to print intermediate outputs
            
        Returns:
            Dictionary containing captions, transcript, and summary
        """
        logger.info(f"Processing video: {video_path}")
        
        # Step 1: Extract frames and audio
        if verbose:
            print("\n[1/4] Extracting frames and audio...")
        frames, audio, sample_rate = self.video_processor.process_video(video_path)
        
        # Step 2: Generate captions
        if verbose:
            print("[2/4] Generating frame captions with BLIP-2...")
        captions_list = self.caption_model.generate_captions_batch(frames)
        captions_text = self.caption_model.combine_captions(captions_list)
        
        if verbose:
            print(f"\nGenerated {len(captions_list)} captions:")
            for i, caption in enumerate(captions_list[:3]):  # Show first 3
                print(f"  Frame {i+1}: {caption}")
            if len(captions_list) > 3:
                print(f"  ... and {len(captions_list) - 3} more frames")
        
        # Step 3: Transcribe audio
        if verbose:
            print("\n[3/4] Transcribing audio with Whisper...")
        transcript = self.transcription_model.get_transcript_text(audio, sample_rate)
        
        if verbose:
            print(f"\nTranscript ({len(transcript)} chars):")
            print(f"  {transcript[:200]}..." if len(transcript) > 200 else f"  {transcript}")
        
        # Step 4: Combine and summarize
        if verbose:
            print("\n[4/4] Generating summary with T5...")
        
        combined_text = f"Video captions: {captions_text} Audio transcript: {transcript}"
        
        summary = self.summarization_model.summarize(
            text=combined_text,
            num_beams=self.config["inference"]["beam_size"],
            temperature=self.config["inference"]["temperature"],
            top_p=self.config["inference"]["top_p"],
            repetition_penalty=self.config["inference"]["repetition_penalty"]
        )
        
        if verbose:
            print("\n" + "="*80)
            print("FINAL SUMMARY:")
            print("="*80)
            print(summary)
            print("="*80 + "\n")
        
        return {
            "video_path": video_path,
            "num_frames": len(frames),
            "captions": captions_text,
            "transcript": transcript,
            "summary": summary
        }


def main():
    parser = argparse.ArgumentParser(description="Generate video summary")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to fine-tuned model checkpoint")
    parser.add_argument("--output", type=str, default=None, help="Output file for summary (optional)")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize summarizer
    summarizer = VideoSummarizer(config, checkpoint_path=args.checkpoint)
    
    # Generate summary
    result = summarizer.summarize_video(args.video, verbose=not args.quiet)
    
    # Save output if requested
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()