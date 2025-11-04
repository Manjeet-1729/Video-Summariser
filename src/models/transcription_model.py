"""
Whisper audio transcription model.
"""
import whisper
import numpy as np
import logging
from typing import Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TranscriptionModel:
    """Whisper model for audio transcription."""
    
    def __init__(self, model_name: str = "base", device: str = "cuda"):
        """
        Initialize Whisper model.
        
        Args:
            model_name: Whisper model size (tiny, base, small, medium, large)
            device: Device to run model on
        """
        self.device = device
        logger.info(f"Loading Whisper model: {model_name}")
        
        try:
            self.model = whisper.load_model(model_name, device=device)
            logger.info("Whisper model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading Whisper model: {str(e)}")
            raise
    
    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> Dict:
        """
        Transcribe audio to text.
        
        Args:
            audio: Audio array
            sample_rate: Sample rate of audio
            
        Returns:
            Dictionary containing transcription results
        """
        try:
            if len(audio) == 0:
                logger.warning("Empty audio provided")
                return {"text": "", "segments": []}
            
            # Ensure audio is float32 and normalized
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            # Normalize audio to [-1, 1] range if needed
            if audio.max() > 1.0 or audio.min() < -1.0:
                audio = audio / np.max(np.abs(audio))
            
            logger.info("Transcribing audio...")
            result = self.model.transcribe(
                audio,
                verbose=False,
                language="en"  # Can be made configurable
            )
            
            logger.info(f"Transcription complete. Text length: {len(result['text'])} chars")
            return result
            
        except Exception as e:
            logger.error(f"Error transcribing audio: {str(e)}")
            return {"text": "", "segments": []}
    
    def get_transcript_text(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Get clean transcript text from audio.
        
        Args:
            audio: Audio array
            sample_rate: Sample rate of audio
            
        Returns:
            Transcript text string
        """
        result = self.transcribe(audio, sample_rate)
        return result.get("text", "").strip()
    
    def get_transcript_with_timestamps(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Get transcript with timestamps.
        
        Args:
            audio: Audio array
            sample_rate: Sample rate of audio
            
        Returns:
            Formatted transcript with timestamps
        """
        result = self.transcribe(audio, sample_rate)
        
        if not result.get("segments"):
            return result.get("text", "")
        
        formatted_segments = []
        for segment in result["segments"]:
            start = segment["start"]
            text = segment["text"].strip()
            formatted_segments.append(f"[{start:.1f}s] {text}")
        
        return " ".join(formatted_segments)