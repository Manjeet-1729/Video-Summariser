"""
Preprocessing script to prepare dataset from videos.
"""
import yaml
import argparse
import logging
from pathlib import Path

from src.utils.video_processor import VideoProcessor
from src.models.caption_model import CaptionModel
from src.models.transcription_model import TranscriptionModel
from src.data.dataset import prepare_dataset_from_videos

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Preprocess videos for training")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--split", type=str, required=True, choices=["train", "val", "test"],
                        help="Dataset split to process")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing videos")
    parser.add_argument("--summaries_file", type=str, required=True,
                        help="JSON file with video summaries")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Processing {args.split} split")
    logger.info(f"Video directory: {args.video_dir}")
    logger.info(f"Summaries file: {args.summaries_file}")
    
    # Initialize models
    logger.info("Initializing models...")
    
    video_processor = VideoProcessor(
        fps=config["video_processing"]["frame_extraction"]["fps"],
        max_frames=config["video_processing"]["frame_extraction"]["max_frames"],
        sample_rate=config["video_processing"]["audio_extraction"]["sample_rate"]
    )
    
    caption_model = CaptionModel(
        model_name=config["models"]["blip2"]["model_name"],
        device=config["models"]["blip2"]["device"]
    )
    
    transcription_model = TranscriptionModel(
        model_name=config["models"]["whisper"]["model_name"],
        device=config["models"]["whisper"]["device"]
    )
    
    # Determine output path
    if args.split == "train":
        output_dir = config["data"]["train_data_path"]
    elif args.split == "val":
        output_dir = config["data"]["val_data_path"]
    else:
        output_dir = config["data"]["test_data_path"]
    
    output_file = Path(output_dir) / "processed_data.jsonl"
    
    # Process videos
    logger.info("Processing videos...")
    prepare_dataset_from_videos(
        video_dir=args.video_dir,
        summaries_file=args.summaries_file,
        output_file=str(output_file),
        caption_model=caption_model,
        transcription_model=transcription_model,
        video_processor=video_processor
    )
    
    logger.info(f"Preprocessing complete! Data saved to {output_file}")


if __name__ == "__main__":
    main()