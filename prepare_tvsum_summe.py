"""
Prepare TVSum and SumMe datasets for video summarization training.
Converts annotations to our format and creates train/val/test splits.
"""
import argparse
import json
import logging
import shutil
from pathlib import Path
import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import h5py

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TVSumSumMePreparator:
    """Prepare TVSum and SumMe datasets."""
    
    def __init__(self, input_dir: str, output_dir: str):
        """
        Initialize preparator.
        
        Args:
            input_dir: Directory containing raw datasets
            output_dir: Directory to save processed datasets
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_tvsum_annotations(self):
        """Load TVSum annotations from MATLAB file."""
        logger.info("Loading TVSum annotations...")
        
        tvsum_dir = self.input_dir / 'tvsum'
        mat_file = tvsum_dir / 'data' / 'ydata-tvsum50.mat'
        
        if not mat_file.exists():
            # Try alternative locations
            alt_paths = [
                tvsum_dir / 'ydata-tvsum50.mat',
                tvsum_dir / 'data' / 'ydata-tvsum50-v1_1.mat',
            ]
            for alt_path in alt_paths:
                if alt_path.exists():
                    mat_file = alt_path
                    break
        
        if not mat_file.exists():
            raise FileNotFoundError(f"TVSum annotation file not found: {mat_file}")
        
        # Load MATLAB file
        data = sio.loadmat(str(mat_file))
        
        # Parse annotations
        videos_data = []
        
        # TVSum structure: video_id, category, title, length, nframes, user_anno
        for i in range(len(data['tvsum50'])):
            video_info = data['tvsum50'][i][0]
            
            video_data = {
                'video_id': str(video_info[0][0]),
                'category': str(video_info[1][0]),
                'title': str(video_info[2][0]) if len(video_info[2]) > 0 else '',
                'length': int(video_info[3][0][0]),
                'nframes': int(video_info[4][0][0]),
                'annotations': video_info[5]  # Frame-level importance scores
            }
            
            # Average annotations from multiple users
            importance_scores = np.mean(video_info[5], axis=0).tolist()
            video_data['importance_scores'] = importance_scores
            
            videos_data.append(video_data)
        
        logger.info(f"Loaded {len(videos_data)} TVSum videos")
        return videos_data
    
    def load_summe_annotations(self):
        """Load SumMe annotations from MATLAB files."""
        logger.info("Loading SumMe annotations...")
        
        summe_dir = self.input_dir / 'summe'
        gt_dir = summe_dir / 'GT'
        
        if not gt_dir.exists():
            # Try alternative location
            gt_dir = summe_dir / 'data' / 'GT'
        
        if not gt_dir.exists():
            raise FileNotFoundError(f"SumMe GT directory not found: {gt_dir}")
        
        videos_data = []
        
        # Process each annotation file
        for mat_file in sorted(gt_dir.glob('*.mat')):
            try:
                data = sio.loadmat(str(mat_file))
                
                video_id = mat_file.stem
                
                # Extract user summaries and scores
                user_summaries = data.get('user_summary', [])
                gt_score = data.get('gt_score', np.array([]))
                
                video_data = {
                    'video_id': video_id,
                    'user_summaries': user_summaries.tolist() if isinstance(user_summaries, np.ndarray) else [],
                    'importance_scores': gt_score.flatten().tolist() if len(gt_score) > 0 else []
                }
                
                videos_data.append(video_data)
                
            except Exception as e:
                logger.warning(f"Error loading {mat_file}: {str(e)}")
                continue
        
        logger.info(f"Loaded {len(videos_data)} SumMe videos")
        return videos_data
    
    def generate_summary_from_scores(self, importance_scores, video_id, category=None):
        """
        Generate textual summary from importance scores.
        
        Args:
            importance_scores: Frame-level importance scores
            video_id: Video identifier
            category: Video category (optional)
            
        Returns:
            Generated summary text
        """
        # For now, generate a template summary
        # This will be replaced by actual BLIP-2 captions during preprocessing
        
        scores_array = np.array(importance_scores)
        avg_importance = np.mean(scores_array)
        max_importance = np.max(scores_array)
        
        # Find peaks (important moments)
        threshold = np.percentile(scores_array, 75)  # Top 25% frames
        important_frames = np.where(scores_array > threshold)[0]
        
        # Generate template summary
        summary_parts = []
        
        if category:
            summary_parts.append(f"This {category.lower()} video")
        else:
            summary_parts.append("This video")
        
        summary_parts.append(f"contains {len(important_frames)} key moments")
        summary_parts.append(f"with an average importance score of {avg_importance:.2f}.")
        
        summary = " ".join(summary_parts)
        
        # Add placeholder for actual captions
        summary += " [To be replaced with BLIP-2 captions and Whisper transcripts during preprocessing]"
        
        return summary
    
    def process_tvsum(self):
        """Process TVSum dataset."""
        logger.info("\n" + "="*60)
        logger.info("Processing TVSum Dataset")
        logger.info("="*60)
        
        # Load annotations
        videos_data = self.load_tvsum_annotations()
        
        # Find video files
        tvsum_dir = self.input_dir / 'tvsum'
        video_dir = tvsum_dir / 'video'
        
        if not video_dir.exists():
            raise FileNotFoundError(f"TVSum video directory not found: {video_dir}")
        
        # Process each video
        processed_data = []
        
        for video_data in tqdm(videos_data, desc="Processing TVSum videos"):
            video_id = video_data['video_id']
            
            # Find video file
            video_files = list(video_dir.glob(f"{video_id}.*"))
            if not video_files:
                logger.warning(f"Video file not found for {video_id}")
                continue
            
            video_path = video_files[0]
            
            # Generate summary
            summary = self.generate_summary_from_scores(
                video_data['importance_scores'],
                video_id,
                video_data.get('category')
            )
            
            processed_entry = {
                'video_id': f"tvsum_{video_id}",
                'video_path': str(video_path),
                'summary': summary,
                'category': video_data.get('category', 'unknown'),
                'duration': video_data['length'],
                'nframes': video_data['nframes'],
                'importance_scores': video_data['importance_scores'],
                'dataset': 'tvsum'
            }
            
            processed_data.append(processed_entry)
        
        logger.info(f"Processed {len(processed_data)} TVSum videos")
        return processed_data
    
    def process_summe(self):
        """Process SumMe dataset."""
        logger.info("\n" + "="*60)
        logger.info("Processing SumMe Dataset")
        logger.info("="*60)
        
        # Load annotations
        videos_data = self.load_summe_annotations()
        
        # Find video files
        summe_dir = self.input_dir / 'summe'
        video_dir = summe_dir / 'videos'
        
        if not video_dir.exists():
            raise FileNotFoundError(f"SumMe video directory not found: {video_dir}")
        
        # Process each video
        processed_data = []
        
        for video_data in tqdm(videos_data, desc="Processing SumMe videos"):
            video_id = video_data['video_id']
            
            # Find video file
            video_files = list(video_dir.glob(f"{video_id}.*"))
            if not video_files:
                logger.warning(f"Video file not found for {video_id}")
                continue
            
            video_path = video_files[0]
            
            # Generate summary
            if video_data['importance_scores']:
                summary = self.generate_summary_from_scores(
                    video_data['importance_scores'],
                    video_id
                )
            else:
                summary = f"This video contains important moments. [To be replaced with captions]"
            
            processed_entry = {
                'video_id': f"summe_{video_id}",
                'video_path': str(video_path),
                'summary': summary,
                'importance_scores': video_data['importance_scores'],
                'user_summaries': video_data['user_summaries'],
                'dataset': 'summe'
            }
            
            processed_data.append(processed_entry)
        
        logger.info(f"Processed {len(processed_data)} SumMe videos")
        return processed_data
    
    def create_splits(self, data, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42):
        """
        Create train/val/test splits.
        
        Args:
            data: List of processed video data
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            seed: Random seed
            
        Returns:
            Dictionary with train/val/test splits
        """
        logger.info("\n" + "="*60)
        logger.info("Creating Data Splits")
        logger.info("="*60)
        
        # Shuffle data
        np.random.seed(seed)
        indices = np.random.permutation(len(data))
        data_shuffled = [data[i] for i in indices]
        
        # Calculate split sizes
        n_total = len(data)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        # Split data
        train_data = data_shuffled[:n_train]
        val_data = data_shuffled[n_train:n_train + n_val]
        test_data = data_shuffled[n_train + n_val:]
        
        logger.info(f"Total videos: {n_total}")
        logger.info(f"Training: {len(train_data)} ({len(train_data)/n_total*100:.1f}%)")
        logger.info(f"Validation: {len(val_data)} ({len(val_data)/n_total*100:.1f}%)")
        logger.info(f"Test: {len(test_data)} ({len(test_data)/n_total*100:.1f}%)")
        
        return {
            'train': train_data,
            'val': val_data,
            'test': test_data
        }
    
    def save_splits(self, splits):
        """Save data splits to output directory."""
        logger.info("\n" + "="*60)
        logger.info("Saving Data Splits")
        logger.info("="*60)
        
        for split_name, split_data in splits.items():
            split_dir = self.output_dir / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            
            # Create videos directory
            videos_dir = split_dir / 'videos'
            videos_dir.mkdir(exist_ok=True)
            
            # Copy videos and create summaries
            summaries = {}
            
            for entry in tqdm(split_data, desc=f"Saving {split_name} split"):
                video_id = entry['video_id']
                src_video = Path(entry['video_path'])
                
                if src_video.exists():
                    # Copy video to split directory
                    dst_video = videos_dir / f"{video_id}{src_video.suffix}"
                    shutil.copy2(src_video, dst_video)
                    
                    # Add to summaries
                    summaries[video_id] = entry['summary']
                else:
                    logger.warning(f"Video file not found: {src_video}")
            
            # Save summaries.json
            summaries_file = split_dir / 'summaries.json'
            with open(summaries_file, 'w') as f:
                json.dump(summaries, f, indent=2)
            
            # Save detailed metadata
            metadata_file = split_dir / 'metadata.json'
            with open(metadata_file, 'w') as f:
                json.dump(split_data, f, indent=2)
            
            logger.info(f"✓ Saved {split_name} split: {len(split_data)} videos")
            logger.info(f"  Location: {split_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare TVSum and SumMe datasets for training"
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input directory containing raw datasets'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='data/processed',
        help='Output directory for processed datasets'
    )
    
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.6,
        help='Training set ratio (default: 0.6)'
    )
    
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.2,
        help='Validation set ratio (default: 0.2)'
    )
    
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.2,
        help='Test set ratio (default: 0.2)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for splitting (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Validate ratios
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 0.01:
        raise ValueError("Train, val, and test ratios must sum to 1.0")
    
    # Create preparator
    preparator = TVSumSumMePreparator(args.input, args.output)
    
    # Process datasets
    try:
        tvsum_data = preparator.process_tvsum()
    except Exception as e:
        logger.error(f"Error processing TVSum: {str(e)}")
        tvsum_data = []
    
    try:
        summe_data = preparator.process_summe()
    except Exception as e:
        logger.error(f"Error processing SumMe: {str(e)}")
        summe_data = []
    
    # Combine datasets
    all_data = tvsum_data + summe_data
    
    if not all_data:
        logger.error("No videos processed! Check your input directory.")
        return
    
    # Create splits
    splits = preparator.create_splits(
        all_data,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
    
    # Save splits
    preparator.save_splits(splits)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("Processing Complete!")
    logger.info("="*60)
    logger.info(f"\nOutput directory: {args.output}")
    logger.info("\nNext steps:")
    logger.info("  1. Run preprocessing:")
    logger.info(f"     python preprocess.py --split train \\")
    logger.info(f"       --video_dir {args.output}/train/videos \\")
    logger.info(f"       --summaries_file {args.output}/train/summaries.json")
    logger.info("  2. Repeat for val and test splits")
    logger.info("  3. Run training: python train.py")


if __name__ == "__main__":
    main()