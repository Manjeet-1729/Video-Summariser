"""
BLIP-2 caption generation model.
"""
import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from typing import List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CaptionModel:
    """BLIP-2 model for generating image captions."""
    
    def __init__(self, model_name: str = "Salesforce/blip2-opt-2.7b", device: str = "cuda"):
        """
        Initialize BLIP-2 model.
        
        Args:
            model_name: Hugging Face model name
            device: Device to run model on
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading BLIP-2 model on {self.device}")
        
        try:
            self.processor = Blip2Processor.from_pretrained(model_name)
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("BLIP-2 model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading BLIP-2 model: {str(e)}")
            raise
    
    def generate_caption(self, image: Image.Image, prompt: str = "a photo of") -> str:
        """
        Generate caption for a single image.
        
        Args:
            image: PIL Image
            prompt: Optional prompt to guide caption generation
            
        Returns:
            Generated caption string
        """
        try:
            inputs = self.processor(images=image, text=prompt, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_length=50,
                    num_beams=3,
                    temperature=1.0
                )
            
            caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            return caption
            
        except Exception as e:
            logger.error(f"Error generating caption: {str(e)}")
            return ""
    
    def generate_captions_batch(self, images: List[Image.Image], prompt: str = "a photo of") -> List[str]:
        """
        Generate captions for multiple images.
        
        Args:
            images: List of PIL Images
            prompt: Optional prompt to guide caption generation
            
        Returns:
            List of caption strings
        """
        captions = []
        
        for i, image in enumerate(images):
            logger.info(f"Generating caption for frame {i+1}/{len(images)}")
            caption = self.generate_caption(image, prompt)
            captions.append(caption)
        
        return captions
    
    def combine_captions(self, captions: List[str]) -> str:
        """
        Combine frame captions into a single text.
        
        Args:
            captions: List of caption strings
            
        Returns:
            Combined caption text
        """
        # Add frame numbers for context
        combined = []
        for i, caption in enumerate(captions):
            if caption:  # Only include non-empty captions
                combined.append(f"Frame {i+1}: {caption}")
        
        return " ".join(combined)