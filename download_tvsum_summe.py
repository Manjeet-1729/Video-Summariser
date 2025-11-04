"""
Download TVSum and SumMe datasets for video summarization.
"""
import argparse
import os
import logging
import requests
import zipfile
import tarfile
from pathlib import Path
from tqdm import tqdm
import subprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TVSumSumMeDownloader:
    """Download and setup TVSum and SumMe datasets."""
    
    DATASETS_INFO = {
        'tvsum': {
            'name': 'TVSum',
            'videos': 50,
            'size': '~2 GB',
            'github': 'https://github.com/yalesong/tvsum',
            'download_instructions': 'Available via GitHub repository'
        },
        'summe': {
            'name': 'SumMe',
            'videos': 25,
            'size': '~1 GB',
            'github': 'https://github.com/gyglim/gm_submodular',
            'download_instructions': 'Videos available in repository'
        }
    }
    
    def __init__(self, output_dir: str):
        """
        Initialize downloader.
        
        Args:
            output_dir: Directory to save datasets
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def download_file(self, url: str, destination: Path, description: str = None):
        """
        Download file with progress bar.
        
        Args:
            url: URL to download from
            destination: Destination file path
            description: Description for progress bar
        """
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(destination, 'wb') as f, tqdm(
                desc=description or f"Downloading {destination.name}",
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    pbar.update(size)
            
            logger.info(f"Downloaded: {destination}")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading {url}: {str(e)}")
            return False
    
    def clone_repo(self, repo_url: str, destination: Path):
        """
        Clone GitHub repository.
        
        Args:
            repo_url: GitHub repository URL
            destination: Destination directory
        """
        try:
            if destination.exists():
                logger.info(f"Repository already exists: {destination}")
                return True
            
            logger.info(f"Cloning repository: {repo_url}")
            subprocess.run(
                ['git', 'clone', repo_url, str(destination)],
                check=True,
                capture_output=True
            )
            logger.info(f"Cloned successfully: {destination}")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error cloning repository: {str(e)}")
            return False
        except FileNotFoundError:
            logger.error("Git not found. Please install git.")
            return False
    
    def download_tvsum(self):
        """Download TVSum dataset."""
        logger.info("\n" + "="*60)
        logger.info("Downloading TVSum Dataset")
        logger.info("="*60)
        
        tvsum_dir = self.output_dir / 'tvsum'
        
        # Clone repository
        if not self.clone_repo(
            'https://github.com/yalesong/tvsum.git',
            tvsum_dir
        ):
            logger.error("Failed to clone TVSum repository")
            return False
        
        # Check if videos exist
        video_dir = tvsum_dir / 'video'
        if not video_dir.exists() or not list(video_dir.glob('*.mp4')):
            logger.warning("\n" + "="*60)
            logger.warning("TVSum videos not found in repository!")
            logger.warning("="*60)
            logger.warning("\nYou need to download videos manually:")
            logger.warning("1. Visit: https://github.com/yalesong/tvsum")
            logger.warning("2. Follow instructions to download videos")
            logger.warning("3. Place videos in: " + str(video_dir))
            logger.warning("\nNote: Videos are hosted on external servers")
            logger.warning("due to licensing restrictions.")
            return False
        
        logger.info(f"✓ TVSum downloaded successfully: {tvsum_dir}")
        return True
    
    def download_summe(self):
        """Download SumMe dataset."""
        logger.info("\n" + "="*60)
        logger.info("Downloading SumMe Dataset")
        logger.info("="*60)
        
        summe_dir = self.output_dir / 'summe'
        
        # Clone repository
        if not self.clone_repo(
            'https://github.com/gyglim/gm_submodular.git',
            summe_dir
        ):
            logger.error("Failed to clone SumMe repository")
            return False
        
        # Check if videos exist
        video_dir = summe_dir / 'videos'
        if not video_dir.exists() or not list(video_dir.glob('*.webm')):
            logger.warning("\n" + "="*60)
            logger.warning("SumMe videos not found in repository!")
            logger.warning("="*60)
            logger.warning("\nYou need to download videos manually:")
            logger.warning("1. Visit: https://github.com/gyglim/gm_submodular")
            logger.warning("2. Download videos from the data/ directory")
            logger.warning("3. Or request access from dataset authors")
            logger.warning("4. Place videos in: " + str(video_dir))
            return False
        
        logger.info(f"✓ SumMe downloaded successfully: {summe_dir}")
        return True
    
    def download_alternative_sources(self):
        """Download from alternative sources (if available)."""
        logger.info("\n" + "="*60)
        logger.info("Alternative Download Methods")
        logger.info("="*60)
        
        logger.info("\nTVSum:")
        logger.info("  Option 1: Visit https://github.com/yalesong/tvsum")
        logger.info("  Option 2: Request from authors at Yale University")
        logger.info("  Option 3: Check academic mirror sites")
        
        logger.info("\nSumMe:")
        logger.info("  Option 1: Visit https://github.com/gyglim/gm_submodular")
        logger.info("  Option 2: ETH Zurich dataset page")
        logger.info("  Option 3: Request from dataset authors")
        
        logger.info("\nBoth datasets require manual download due to:")
        logger.info("  - Copyright/licensing restrictions")
        logger.info("  - Large file sizes")
        logger.info("  - Academic usage agreements")
    
    def verify_downloads(self):
        """Verify downloaded datasets."""
        logger.info("\n" + "="*60)
        logger.info("Verifying Downloads")
        logger.info("="*60)
        
        results = {}
        
        # Check TVSum
        tvsum_dir = self.output_dir / 'tvsum'
        tvsum_videos = list((tvsum_dir / 'video').glob('*.mp4')) if (tvsum_dir / 'video').exists() else []
        results['tvsum'] = {
            'downloaded': tvsum_dir.exists(),
            'videos': len(tvsum_videos),
            'expected_videos': 50
        }
        
        # Check SumMe
        summe_dir = self.output_dir / 'summe'
        summe_videos = list((summe_dir / 'videos').glob('*.webm')) if (summe_dir / 'videos').exists() else []
        results['summe'] = {
            'downloaded': summe_dir.exists(),
            'videos': len(summe_videos),
            'expected_videos': 25
        }
        
        # Print results
        for dataset, info in results.items():
            status = "✓" if info['videos'] == info['expected_videos'] else "✗"
            logger.info(f"\n{status} {dataset.upper()}:")
            logger.info(f"  Repository: {'Yes' if info['downloaded'] else 'No'}")
            logger.info(f"  Videos: {info['videos']}/{info['expected_videos']}")
            
            if info['videos'] < info['expected_videos']:
                logger.warning(f"  Missing {info['expected_videos'] - info['videos']} videos!")
        
        return results
    
    def print_summary(self):
        """Print download summary and next steps."""
        logger.info("\n" + "="*60)
        logger.info("Download Summary")
        logger.info("="*60)
        
        logger.info("\nDatasets Location:")
        logger.info(f"  {self.output_dir.absolute()}")
        
        logger.info("\nNext Steps:")
        logger.info("  1. Verify videos are downloaded correctly")
        logger.info("  2. Run: python prepare_tvsum_summe.py \\")
        logger.info("            --input " + str(self.output_dir) + " \\")
        logger.info("            --output data/processed")
        logger.info("  3. This will convert datasets to our format")
        logger.info("  4. Then run preprocessing: python preprocess.py")
        
        logger.info("\nIf videos are missing:")
        logger.info("  - Follow manual download instructions above")
        logger.info("  - Place videos in respective directories")
        logger.info("  - Run this script again to verify")


def main():
    parser = argparse.ArgumentParser(
        description="Download TVSum and SumMe datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download both datasets
  python download_tvsum_summe.py --output data/raw

  # Download only TVSum
  python download_tvsum_summe.py --output data/raw --dataset tvsum

  # Download only SumMe
  python download_tvsum_summe.py --output data/raw --dataset summe

Note: These datasets require manual video download due to licensing.
This script will clone the repositories and guide you through the process.
        """
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='data/raw',
        help='Output directory for datasets (default: data/raw)'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['tvsum', 'summe', 'both'],
        default='both',
        help='Which dataset to download (default: both)'
    )
    
    args = parser.parse_args()
    
    # Create downloader
    downloader = TVSumSumMeDownloader(args.output)
    
    # Download datasets
    if args.dataset in ['tvsum', 'both']:
        downloader.download_tvsum()
    
    if args.dataset in ['summe', 'both']:
        downloader.download_summe()
    
    # Show alternative download methods
    downloader.download_alternative_sources()
    
    # Verify downloads
    downloader.verify_downloads()
    
    # Print summary
    downloader.print_summary()


if __name__ == "__main__":
    main()