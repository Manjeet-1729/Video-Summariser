"""
Dataset preparation and loading utilities.
"""
import torch
from torch.utils.data import Dataset, DataLoader
import json
import os
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoSummaryDataset(Dataset):
    """Dataset for video summarization task."""
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_input_length: int = 1024,
        max_output_length: int = 256
    ):
        """
        Initialize dataset.
        
        Args:
            data_path: Path to JSONL file containing processed data
            tokenizer: T5 tokenizer
            max_input_length: Maximum input sequence length
            max_output_length: Maximum output sequence length
        """
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        
        # Load data
        self.data = []
        logger.info(f"Loading data from {data_path}")
        
        if os.path.exists(data_path):
            with open(data_path, 'r') as f:
                for line in f:
                    self.data.append(json.loads(line))
            logger.info(f"Loaded {len(self.data)} samples")
        else:
            logger.warning(f"Data file not found: {data_path}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single training sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing tokenized inputs and labels
        """
        sample = self.data[idx]
        
        # Combine captions and transcript
        captions = sample.get("captions", "")
        transcript = sample.get("transcript", "")
        combined_text = f"Video captions: {captions} Audio transcript: {transcript}"
        
        # Get target summary
        summary = sample.get("summary", "")
        
        # Tokenize input
        input_text = f"summarize: {combined_text}"
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Tokenize target
        target_encoding = self.tokenizer(
            summary,
            max_length=self.max_output_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Prepare labels (replace padding token id with -100)
        labels = target_encoding["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": input_encoding["input_ids"].squeeze(),
            "attention_mask": input_encoding["attention_mask"].squeeze(),
            "labels": labels.squeeze()
        }


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 4
) -> DataLoader:
    """
    Create DataLoader from dataset.
    
    Args:
        dataset: VideoSummaryDataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        
    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )


def prepare_dataset_from_videos(
    video_dir: str,
    summaries_file: str,
    output_file: str,
    caption_model,
    transcription_model,
    video_processor
):
    """
    Process videos and create dataset file.
    
    Args:
        video_dir: Directory containing video files
        summaries_file: JSON file with video_id -> summary mapping
        output_file: Output JSONL file path
        caption_model: CaptionModel instance
        transcription_model: TranscriptionModel instance
        video_processor: VideoProcessor instance
    """
    # Load summaries
    with open(summaries_file, 'r') as f:
        summaries = json.load(f)
    
    # Process each video
    processed_data = []
    video_files = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
    
    logger.info(f"Processing {len(video_files)} videos...")
    
    for video_file in video_files:
        video_id = os.path.splitext(video_file)[0]
        video_path = os.path.join(video_dir, video_file)
        
        if video_id not in summaries:
            logger.warning(f"No summary found for {video_id}, skipping")
            continue
        
        try:
            # Extract frames and audio
            frames, audio, sr = video_processor.process_video(video_path)
            
            # Generate captions
            captions_list = caption_model.generate_captions_batch(frames)
            captions_text = caption_model.combine_captions(captions_list)
            
            # Generate transcript
            transcript = transcription_model.get_transcript_text(audio, sr)
            
            # Create sample
            sample = {
                "video_id": video_id,
                "captions": captions_text,
                "transcript": transcript,
                "summary": summaries[video_id]
            }
            
            processed_data.append(sample)
            logger.info(f"Processed {video_id} ({len(processed_data)}/{len(video_files)})")
            
        except Exception as e:
            logger.error(f"Error processing {video_id}: {str(e)}")
            continue
    
    # Save to file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        for sample in processed_data:
            f.write(json.dumps(sample) + '\n')
    
    logger.info(f"Saved {len(processed_data)} samples to {output_file}")