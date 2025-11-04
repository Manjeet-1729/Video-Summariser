"""
Analyze TVSum and SumMe datasets.
Shows statistics, distributions, and sample visualizations.
"""
import argparse
import json
import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import cv2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetAnalyzer:
    """Analyze video summarization datasets."""
    
    def __init__(self, data_dir: str):
        """
        Initialize analyzer.
        
        Args:
            data_dir: Directory containing processed data
        """
        self.data_dir = Path(data_dir)
    
    def load_split_data(self, split: str):
        """Load data for a specific split."""
        metadata_file = self.data_dir / split / 'metadata.json'
        
        if not metadata_file.exists():
            logger.warning(f"Metadata not found for {split} split")
            return []
        
        with open(metadata_file, 'r') as f:
            data = json.load(f)
        
        return data
    
    def analyze_basic_stats(self):
        """Analyze basic dataset statistics."""
        logger.info("\n" + "="*70)
        logger.info("BASIC DATASET STATISTICS")
        logger.info("="*70)
        
        splits = ['train', 'val', 'test']
        total_videos = 0
        
        for split in splits:
            data = self.load_split_data(split)
            total_videos += len(data)
            
            if data:
                logger.info(f"\n{split.upper()} Split:")
                logger.info(f"  Videos: {len(data)}")
                
                # Count by dataset
                datasets = Counter([v.get('dataset', 'unknown') for v in data])
                for dataset, count in datasets.items():
                    logger.info(f"    {dataset.upper()}: {count} videos")
        
        logger.info(f"\nTotal Videos: {total_videos}")
    
    def analyze_video_properties(self):
        """Analyze video properties (duration, frames, etc.)."""
        logger.info("\n" + "="*70)
        logger.info("VIDEO PROPERTIES")
        logger.info("="*70)
        
        all_data = []
        for split in ['train', 'val', 'test']:
            all_data.extend(self.load_split_data(split))
        
        if not all_data:
            logger.warning("No data to analyze")
            return
        
        # Extract properties
        durations = [v.get('duration', 0) for v in all_data if 'duration' in v]
        nframes = [v.get('nframes', 0) for v in all_data if 'nframes' in v]
        
        if durations:
            logger.info("\nDuration (seconds):")
            logger.info(f"  Min: {min(durations):.1f}s")
            logger.info(f"  Max: {max(durations):.1f}s")
            logger.info(f"  Mean: {np.mean(durations):.1f}s")
            logger.info(f"  Median: {np.median(durations):.1f}s")
            logger.info(f"  Total: {sum(durations)/3600:.1f} hours")
        
        if nframes:
            logger.info("\nNumber of Frames:")
            logger.info(f"  Min: {min(nframes)}")
            logger.info(f"  Max: {max(nframes)}")
            logger.info(f"  Mean: {int(np.mean(nframes))}")
            logger.info(f"  Median: {int(np.median(nframes))}")
    
    def analyze_categories(self):
        """Analyze video categories (TVSum)."""
        logger.info("\n" + "="*70)
        logger.info("VIDEO CATEGORIES")
        logger.info("="*70)
        
        all_data = []
        for split in ['train', 'val', 'test']:
            all_data.extend(self.load_split_data(split))
        
        # Count categories
        categories = Counter([v.get('category', 'unknown') for v in all_data if 'category' in v])
        
        if categories:
            logger.info("\nCategory Distribution:")
            for category, count in categories.most_common():
                percentage = count / len(all_data) * 100
                logger.info(f"  {category:20s}: {count:3d} videos ({percentage:5.1f}%)")
    
    def analyze_importance_scores(self):
        """Analyze importance score distributions."""
        logger.info("\n" + "="*70)
        logger.info("IMPORTANCE SCORE ANALYSIS")
        logger.info("="*70)
        
        all_data = []
        for split in ['train', 'val', 'test']:
            all_data.extend(self.load_split_data(split))
        
        # Collect all importance scores
        all_scores = []
        for video in all_data:
            scores = video.get('importance_scores', [])
            if scores:
                all_scores.extend(scores)
        
        if all_scores:
            scores_array = np.array(all_scores)
            
            logger.info("\nImportance Scores:")
            logger.info(f"  Min: {scores_array.min():.3f}")
            logger.info(f"  Max: {scores_array.max():.3f}")
            logger.info(f"  Mean: {scores_array.mean():.3f}")
            logger.info(f"  Std: {scores_array.std():.3f}")
            logger.info(f"  Median: {np.median(scores_array):.3f}")
            
            # Percentiles
            logger.info("\nPercentiles:")
            for p in [25, 50, 75, 90, 95]:
                val = np.percentile(scores_array, p)
                logger.info(f"  {p}th: {val:.3f}")
    
    def visualize_distributions(self, save_dir: str = None):
        """Create visualization plots."""
        logger.info("\n" + "="*70)
        logger.info("GENERATING VISUALIZATIONS")
        logger.info("="*70)
        
        all_data = []
        for split in ['train', 'val', 'test']:
            all_data.extend(self.load_split_data(split))
        
        if not all_data:
            logger.warning("No data to visualize")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Dataset Analysis - TVSum & SumMe', fontsize=16, fontweight='bold')
        
        # 1. Duration distribution
        durations = [v.get('duration', 0) for v in all_data if 'duration' in v]
        if durations:
            axes[0, 0].hist(durations, bins=20, color='skyblue', edgecolor='black')
            axes[0, 0].set_xlabel('Duration (seconds)')
            axes[0, 0].set_ylabel('Number of Videos')
            axes[0, 0].set_title('Video Duration Distribution')
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Category distribution
        categories = Counter([v.get('category', 'unknown') for v in all_data if 'category' in v])
        if categories:
            cats, counts = zip(*categories.most_common())
            axes[0, 1].barh(cats, counts, color='lightcoral')
            axes[0, 1].set_xlabel('Number of Videos')
            axes[0, 1].set_title('Category Distribution')
            axes[0, 1].grid(True, alpha=0.3, axis='x')
        
        # 3. Importance score distribution
        all_scores = []
        for video in all_data:
            scores = video.get('importance_scores', [])
            if scores:
                all_scores.extend(scores)
        
        if all_scores:
            axes[1, 0].hist(all_scores, bins=50, color='lightgreen', edgecolor='black')
            axes[1, 0].set_xlabel('Importance Score')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Frame Importance Score Distribution')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Split distribution
        split_counts = {}
        for split in ['train', 'val', 'test']:
            data = self.load_split_data(split)
            split_counts[split] = len(data)
        
        if split_counts:
            splits, counts = zip(*split_counts.items())
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
            axes[1, 1].pie(counts, labels=splits, autopct='%1.1f%%', colors=colors, startangle=90)
            axes[1, 1].set_title('Train/Val/Test Split')
        
        plt.tight_layout()
        
        # Save or show
        if save_dir:
            save_path = Path(save_dir) / 'dataset_analysis.png'
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"✓ Saved visualization: {save_path}")
        else:
            plt.show()
    
    def show_sample_videos(self, n_samples: int = 3):
        """Show information about sample videos."""
        logger.info("\n" + "="*70)
        logger.info(f"SAMPLE VIDEOS (showing {n_samples} random samples)")
        logger.info("="*70)
        
        all_data = []
        for split in ['train', 'val', 'test']:
            all_data.extend(self.load_split_data(split))
        
        if not all_data:
            logger.warning("No data to sample")
            return
        
        # Random sample
        samples = np.random.choice(all_data, min(n_samples, len(all_data)), replace=False)
        
        for i, video in enumerate(samples, 1):
            logger.info(f"\nSample {i}:")
            logger.info(f"  ID: {video.get('video_id', 'unknown')}")
            logger.info(f"  Dataset: {video.get('dataset', 'unknown')}")
            logger.info(f"  Category: {video.get('category', 'N/A')}")
            logger.info(f"  Duration: {video.get('duration', 0):.1f}s")
            logger.info(f"  Frames: {video.get('nframes', 'N/A')}")
            
            if 'importance_scores' in video and video['importance_scores']:
                scores = np.array(video['importance_scores'])
                logger.info(f"  Avg Importance: {scores.mean():.3f}")
                logger.info(f"  Max Importance: {scores.max():.3f}")
            
            summary = video.get('summary', '')
            if summary:
                summary_preview = summary[:100] + "..." if len(summary) > 100 else summary
                logger.info(f"  Summary: {summary_preview}")
    
    def check_data_quality(self):
        """Check for potential data quality issues."""
        logger.info("\n" + "="*70)
        logger.info("DATA QUALITY CHECK")
        logger.info("="*70)
        
        issues = []
        
        for split in ['train', 'val', 'test']:
            data = self.load_split_data(split)
            
            for video in data:
                video_id = video.get('video_id', 'unknown')
                
                # Check for missing video file
                video_path = Path(video.get('video_path', ''))
                if not video_path.exists():
                    issues.append(f"{split}/{video_id}: Video file not found")
                
                # Check for missing summary
                if not video.get('summary'):
                    issues.append(f"{split}/{video_id}: No summary")
                
                # Check for missing importance scores
                if not video.get('importance_scores'):
                    issues.append(f"{split}/{video_id}: No importance scores")
                
                # Check for zero duration
                if video.get('duration', 0) == 0:
                    issues.append(f"{split}/{video_id}: Zero duration")
        
        if issues:
            logger.warning(f"\nFound {len(issues)} potential issues:")
            for issue in issues[:10]:  # Show first 10
                logger.warning(f"  ⚠ {issue}")
            if len(issues) > 10:
                logger.warning(f"  ... and {len(issues) - 10} more")
        else:
            logger.info("\n✓ No data quality issues found!")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze TVSum and SumMe datasets"
    )
    
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/processed',
        help='Directory containing processed data'
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate visualization plots'
    )
    
    parser.add_argument(
        '--save_plots',
        type=str,
        default=None,
        help='Directory to save plots (if not specified, plots are shown)'
    )
    
    parser.add_argument(
        '--samples',
        type=int,
        default=3,
        help='Number of sample videos to show (default: 3)'
    )
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = DatasetAnalyzer(args.data_dir)
    
    # Run analyses
    analyzer.analyze_basic_stats()
    analyzer.analyze_video_properties()
    analyzer.analyze_categories()
    analyzer.analyze_importance_scores()
    analyzer.show_sample_videos(n_samples=args.samples)
    analyzer.check_data_quality()
    
    # Generate visualizations
    if args.visualize:
        try:
            analyzer.visualize_distributions(save_dir=args.save_plots)
        except Exception as e:
            logger.error(f"Error generating visualizations: {str(e)}")
    
    logger.info("\n" + "="*70)
    logger.info("Analysis Complete!")
    logger.info("="*70)


if __name__ == "__main__":
    main()