"""
Training script for T5 video summarization model.
"""
import torch
import yaml
import os
import argparse
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import wandb
from pathlib import Path

from src.data.dataset import VideoSummaryDataset, create_dataloader
from src.utils.metrics import calculate_rouge_scores

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    """Trainer class for T5 fine-tuning."""
    
    def __init__(self, config: dict):
        """
        Initialize trainer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Get device from config or default to cuda
        model_device = config.get("models", {}).get("t5", {}).get("device", "cuda")
        self.device = torch.device(model_device if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize tokenizer and model
        model_name = config["models"]["t5"]["model_name"]
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)
        
        # Create output directories
        self.output_dir = Path(config["training"]["output_dir"])
        self.checkpoint_dir = Path(config["training"]["checkpoint_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize wandb if enabled
        if config["system"]["use_wandb"]:
            wandb.init(
                project=config["system"]["wandb_project"],
                config=config
            )
    
    def prepare_data(self):
        """Prepare training and validation datasets."""
        logger.info("Preparing datasets...")
        
        # Load datasets
        train_dataset = VideoSummaryDataset(
            data_path=self.config["data"]["train_data_path"] + "/processed_data.jsonl",
            tokenizer=self.tokenizer,
            max_input_length=self.config["models"]["t5"]["max_input_length"],
            max_output_length=self.config["models"]["t5"]["max_output_length"]
        )
        
        val_dataset = VideoSummaryDataset(
            data_path=self.config["data"]["val_data_path"] + "/processed_data.jsonl",
            tokenizer=self.tokenizer,
            max_input_length=self.config["models"]["t5"]["max_input_length"],
            max_output_length=self.config["models"]["t5"]["max_output_length"]
        )
        
        # Create dataloaders
        self.train_loader = create_dataloader(
            train_dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=True,
            num_workers=self.config["system"]["num_workers"]
        )
        
        self.val_loader = create_dataloader(
            val_dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=False,
            num_workers=self.config["system"]["num_workers"]
        )
        
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(val_dataset)}")
    
    def setup_optimizer(self):
        """Setup optimizer and scheduler."""
        num_training_steps = len(self.train_loader) * self.config["training"]["num_epochs"]
        num_training_steps //= self.config["training"]["gradient_accumulation_steps"]
        
        # Ensure learning_rate is float
        learning_rate = float(self.config["training"]["learning_rate"])
        weight_decay = float(self.config["training"]["weight_decay"])
        
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config["training"]["warmup_steps"],
            num_training_steps=num_training_steps
        )
    
    def train_epoch(self, epoch: int):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(**batch)
            loss = outputs.loss
            loss = loss / self.config["training"]["gradient_accumulation_steps"]
            
            # Backward pass
            loss.backward()
            total_loss += loss.item()
            
            # Update weights
            if (step + 1) % self.config["training"]["gradient_accumulation_steps"] == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config["training"]["max_grad_norm"]
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            # Update progress bar
            progress_bar.set_postfix({"loss": loss.item() * self.config["training"]["gradient_accumulation_steps"]})
            
            # Log to wandb
            if self.config["system"]["use_wandb"] and step % self.config["training"]["logging_steps"] == 0:
                wandb.log({
                    "train_loss": loss.item() * self.config["training"]["gradient_accumulation_steps"],
                    "learning_rate": self.scheduler.get_last_lr()[0],
                    "epoch": epoch
                })
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def evaluate(self):
        """Evaluate model on validation set."""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_references = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = self.model(**batch)
                total_loss += outputs.loss.item()
                
                # Generate predictions
                generated_ids = self.model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    max_length=self.config["models"]["t5"]["max_output_length"]
                )
                
                predictions = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                references = self.tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)
                
                all_predictions.extend(predictions)
                all_references.extend(references)
        
        avg_loss = total_loss / len(self.val_loader)
        
        # Calculate ROUGE scores
        rouge_scores = calculate_rouge_scores(all_predictions, all_references)
        
        return avg_loss, rouge_scores
    
    def train(self):
        """Main training loop."""
        self.prepare_data()
        self.setup_optimizer()
        
        best_rouge = 0
        
        for epoch in range(1, self.config["training"]["num_epochs"] + 1):
            logger.info(f"\nEpoch {epoch}/{self.config['training']['num_epochs']}")
            
            # Train
            train_loss = self.train_epoch(epoch)
            logger.info(f"Training loss: {train_loss:.4f}")
            
            # Evaluate
            if epoch % (self.config["training"]["eval_steps"] // len(self.train_loader)) == 0:
                val_loss, rouge_scores = self.evaluate()
                logger.info(f"Validation loss: {val_loss:.4f}")
                logger.info(f"ROUGE scores: {rouge_scores}")
                
                # Log to wandb
                if self.config["system"]["use_wandb"]:
                    wandb.log({
                        "val_loss": val_loss,
                        **rouge_scores,
                        "epoch": epoch
                    })
                
                # Save best model
                if rouge_scores["rouge-l"] > best_rouge:
                    best_rouge = rouge_scores["rouge-l"]
                    self.save_checkpoint("best_model")
                    logger.info(f"Saved best model with ROUGE-L: {best_rouge:.4f}")
            
            # Save periodic checkpoint
            if epoch % (self.config["training"]["save_steps"] // len(self.train_loader)) == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch}")
        
        logger.info("Training completed!")
    
    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
        checkpoint_path = self.checkpoint_dir / name
        self.model.save_pretrained(checkpoint_path)
        self.tokenizer.save_pretrained(checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(description="Train T5 video summarization model")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set seed
    torch.manual_seed(config["system"]["seed"])
    
    # Train
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
