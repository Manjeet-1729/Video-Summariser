"""
Video processing utilities for frame and audio extraction.
"""
import cv2
import os
import numpy as np
from PIL import Image
from moviepy.editor import VideoFileClip
import tempfile
from typing import List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoProcessor:
    """Handles video frame and audio extraction."""
    
    def __init__(self, fps: int = 1, max_frames: int = 50, sample_rate: int = 16000):
        """
        Initialize VideoProcessor.
        
        Args:
            fps: Frames per second to extract
            max_frames: Maximum number of frames to extract
            sample_rate: Audio sample rate
        """
        self.fps = fps
        self.max_frames = max_frames
        self.sample_rate = sample_rate
    
    def extract_frames(self, video_path: str) -> List[Image.Image]:
        """
        Extract frames from video at specified fps.
        
        Args:
            video_path: Path to video file
            
        Returns:
            List of PIL Image objects
        """
        frames = []
        
        try:
            cap = cv2.VideoCapture(video_path)
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Calculate frame interval
            frame_interval = int(video_fps / self.fps)
            
            logger.info(f"Video FPS: {video_fps}, Total frames: {total_frames}")
            logger.info(f"Extracting every {frame_interval}th frame")
            
            frame_count = 0
            extracted_count = 0
            
            while cap.isOpened() and extracted_count < self.max_frames:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                if frame_count % frame_interval == 0:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    frames.append(pil_image)
                    extracted_count += 1
                
                frame_count += 1
            
            cap.release()
            logger.info(f"Extracted {len(frames)} frames from {video_path}")
            
        except Exception as e:
            logger.error(f"Error extracting frames: {str(e)}")
            raise
        
        return frames
    
    def extract_audio(self, video_path: str) -> Tuple[np.ndarray, int]:
        """
        Extract audio from video.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        try:
            video = VideoFileClip(video_path)
            
            if video.audio is None:
                logger.warning(f"No audio track found in {video_path}")
                return np.array([]), self.sample_rate
            
            # Create temporary file for audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
                audio_path = tmp_audio.name
            
            # Extract audio
            video.audio.write_audiofile(
                audio_path,
                fps=self.sample_rate,
                codec='pcm_s16le',
                logger=None  # Suppress moviepy logs
            )
            
            # Read audio file
            import soundfile as sf
            audio_array, sr = sf.read(audio_path)
            
            # Clean up
            os.remove(audio_path)
            video.close()
            
            logger.info(f"Extracted audio from {video_path}, duration: {len(audio_array)/sr:.2f}s")
            
            return audio_array, sr
            
        except Exception as e:
            logger.error(f"Error extracting audio: {str(e)}")
            raise
    
    def process_video(self, video_path: str) -> Tuple[List[Image.Image], np.ndarray, int]:
        """
        Process video to extract both frames and audio.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Tuple of (frames, audio_array, sample_rate)
        """
        logger.info(f"Processing video: {video_path}")
        
        frames = self.extract_frames(video_path)
        audio_array, sample_rate = self.extract_audio(video_path)
        
        return frames, audio_array, sample_rate