"""
T5 summarization model.
"""
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SummarizationModel:
    """T5 model for text summarization."""
    
    def __init__(
        self,
        model_name: str = "t5-base",
        device: str = "cuda",
        max_input_length: int = 1024,
        max_output_length: int = 256
    ):
        """
        Initialize T5 model.
        
        Args:
            model_name: Hugging Face model name
            device: Device to run model on
            max_input_length: Maximum input sequence length
            max_output_length: Maximum output sequence length
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        
        logger.info(f"Loading T5 model: {model_name} on {self.device}")
        
        try:
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("T5 model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading T5 model: {str(e)}")
            raise
    
    def summarize(
        self,
        text: str,
        num_beams: int = 4,
        temperature: float = 1.0,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2
    ) -> str:
        """
        Generate summary from input text.
        
        Args:
            text: Input text to summarize
            num_beams: Number of beams for beam search
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            repetition_penalty: Repetition penalty
            
        Returns:
            Generated summary
        """
        try:
            # Prepare input with T5 prefix
            input_text = f"summarize: {text}"
            
            # Tokenize
            inputs = self.tokenizer(
                input_text,
                max_length=self.max_input_length,
                truncation=True,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate summary
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_length=self.max_output_length,
                    num_beams=num_beams,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    early_stopping=True
                )
            
            # Decode
            summary = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
            logger.info(f"Generated summary of {len(summary)} characters")
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return ""
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load fine-tuned model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint directory
        """
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        try:
            self.model = T5ForConditionalGeneration.from_pretrained(checkpoint_path)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("Checkpoint loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {str(e)}")
            raise