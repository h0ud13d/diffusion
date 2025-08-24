# enhanced_train.py - Enhanced training pipeline for the improved diffusion model
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from typing import Dict, Tuple, Optional, List
import time
from tqdm import tqdm

from src.enhanced_model import EnhancedUNet1D, ConditionEncoder
from src.enhanced_diffusion import EnhancedGaussianDiffusion1D, EnhancedDiffusionConfig, EMAv2
from src.portfolio_framework import PortfolioDataManager, PortfolioDataset

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class EnhancedTrainer:
    """Enhanced training pipeline with advanced features"""
    
    def __init__(self, 
                 model_config: Dict,
                 diffusion_config: EnhancedDiffusionConfig,
                 training_config: Dict,
                 device: str = "auto"):
        
        self.model_config = model_config
        self.diffusion_config = diffusion_config
        self.training_config = training_config
        
        # Set device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        # Initialize models
        self.model = None
        self.condition_encoder = None
        self.diffusion = None
        self.ema = None
        
        # Training state
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.epoch = 0
        self.best_val_loss = float('inf')
        
    def build_models(self, feature_dim: int, condition_dim: int, return_dim: int):
        """Build the enhanced models"""
        
        print("Building enhanced models...")
        
        # Condition encoder
        if condition_dim > 0:
            self.condition_encoder = ConditionEncoder(
                input_dim=condition_dim,
                output_dim=self.model_config.get('context_dim', 128),
                hidden_dim=self.model_config.get('condition_hidden_dim', 256)
            ).to(self.device)
        
        # Main UNet model - for diffusion, input and output should be the same
        self.model = EnhancedUNet1D(
            c_in=return_dim,  # Input: returns only
            c_out=return_dim,  # Output: returns only
            base_channels=self.model_config.get('base_channels', 64),
            channel_multipliers=self.model_config.get('channel_multipliers', [1, 2, 4, 8]),
            time_dim=self.model_config.get('time_dim', 256),
            context_dim=self.model_config.get('context_dim', 128) if condition_dim > 0 else None,
            dropout=self.model_config.get('dropout', 0.1),
            attention_resolutions=self.model_config.get('attention_resolutions', [1, 2]),
            num_heads=self.model_config.get('num_heads', 8),
            use_checkpoint=self.model_config.get('use_checkpoint', False)
        ).to(self.device)
        
        # Diffusion wrapper
        self.diffusion = EnhancedGaussianDiffusion1D(
            self.model, self.diffusion_config
        ).to(self.device)
        
        # EMA
        self.ema = EMAv2(
            self.model,
            decay=self.training_config.get('ema_decay', 0.9999),
            warmup_steps=self.training_config.get('ema_warmup', 1000)
        )
        
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
    def setup_training(self):
        """Setup training components"""
        
        # Optimizer
        lr = self.training_config.get('learning_rate', 1e-4)
        weight_decay = self.training_config.get('weight_decay', 1e-6)
        
        params_to_optimize = list(self.model.parameters())
        if self.condition_encoder is not None:
            params_to_optimize += list(self.condition_encoder.parameters())
        
        self.optimizer = optim.AdamW(params_to_optimize, lr=lr, weight_decay=weight_decay)
        
        # Scheduler
        if self.training_config.get('use_scheduler', True):
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.training_config.get('scheduler_T0', 10),
                T_mult=self.training_config.get('scheduler_Tmult', 2),
                eta_min=lr * 0.01
            )
        
        # Mixed precision scaler
        if self.training_config.get('use_amp', True) and self.device.startswith('cuda'):
            self.scaler = torch.cuda.amp.GradScaler()
        
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        
        self.model.train()
        if self.condition_encoder is not None:
            self.condition_encoder.train()
        
        total_loss = 0.0
        total_samples = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            x = batch['x'].to(self.device, non_blocking=True)  # (B, feature_dim, L)
            conditions = batch['conditions'].to(self.device, non_blocking=True)  # (B, condition_dim)
            
            # Use all features as input, predict only returns
            # x contains all features, model will predict returns
            
            # Encode conditions
            if self.condition_encoder is not None and conditions.numel() > 0:
                context = self.condition_encoder(conditions)
            else:
                context = None
            
            # Zero gradients
            self.optimizer.zero_grad(set_to_none=True)
            
            # Extract returns for training (first few channels are returns)
            returns_dim = self.diffusion.model.c_out
            x_returns = x[:, :returns_dim, :]  # (B, return_dim, L)
            
            # Forward pass with mixed precision
            use_amp = (self.scaler is not None)
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = self.diffusion.loss_on(x_returns, context=context)
            
            # Backward pass
            if use_amp:
                self.scaler.scale(loss).backward()
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(self.model.parameters()) + 
                    (list(self.condition_encoder.parameters()) if self.condition_encoder else []),
                    max_norm=self.training_config.get('grad_clip', 1.0)
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.model.parameters()) + 
                    (list(self.condition_encoder.parameters()) if self.condition_encoder else []),
                    max_norm=self.training_config.get('grad_clip', 1.0)
                )
                self.optimizer.step()
            
            # Update EMA
            self.ema.update(self.model)
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step(epoch + batch_idx / len(train_loader))
            
            # Logging
            batch_size = x.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.6f}",
                'avg_loss': f"{total_loss / total_samples:.6f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
        
        return {
            'train_loss': total_loss / total_samples,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch"""
        
        self.model.eval()
        if self.condition_encoder is not None:
            self.condition_encoder.eval()
        
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Move to device
                x = batch['x'].to(self.device, non_blocking=True)
                conditions = batch['conditions'].to(self.device, non_blocking=True)
                
                # Extract return channels
                returns_dim = self.diffusion.model.c_out
                x_returns = x[:, :returns_dim, :]
                
                # Encode conditions
                if self.condition_encoder is not None and conditions.numel() > 0:
                    context = self.condition_encoder(conditions)
                else:
                    context = None
                
                # Forward pass
                loss = self.diffusion.loss_on(x_returns, context=context)
                
                # Accumulate loss
                batch_size = x.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
        
        return {'val_loss': total_loss / total_samples}
    
    def save_checkpoint(self, filepath: str, metadata: Dict):
        """Save training checkpoint"""
        
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'ema_state_dict': self.ema.shadow,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'model_config': self.model_config,
            'diffusion_config': {
                'timesteps': self.diffusion_config.timesteps,
                'beta_schedule': self.diffusion_config.beta_schedule,
                'v_prediction': self.diffusion_config.v_prediction,
                'classifier_free_guidance': self.diffusion_config.classifier_free_guidance,
                'guidance_scale': self.diffusion_config.guidance_scale,
                'drop_prob': self.diffusion_config.drop_prob
            },
            'training_config': self.training_config,
            'metadata': metadata
        }
        
        if self.condition_encoder is not None:
            checkpoint['condition_encoder_state_dict'] = self.condition_encoder.state_dict()
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filepath: str) -> Dict:
        """Load training checkpoint"""
        
        print(f"Loading checkpoint: {filepath}")
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Load states
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.ema.shadow = checkpoint['ema_state_dict']
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.condition_encoder is not None and 'condition_encoder_state_dict' in checkpoint:
            self.condition_encoder.load_state_dict(checkpoint['condition_encoder_state_dict'])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return checkpoint['metadata']
    
    def train(self,
              train_dataset: PortfolioDataset,
              val_dataset: PortfolioDataset,
              metadata: Dict,
              epochs: int,
              save_path: str = "enhanced_diffusion_model.pt") -> Dict:
        """Main training loop"""
        
        print("Starting enhanced training...")
        
        # Build models
        feature_dim = len(metadata['feature_columns'])
        condition_dim = len(metadata['condition_columns'])
        return_dim = len(metadata['return_columns'])
        
        self.build_models(feature_dim, condition_dim, return_dim)
        self.setup_training()
        
        # Data loaders
        batch_size = self.training_config.get('batch_size', 32)
        num_workers = self.training_config.get('num_workers', 4)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
        # Training loop
        for epoch in range(1, epochs + 1):
            self.epoch = epoch
            start_time = time.time()
            
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_metrics = self.validate_epoch(val_loader)
            
            # Update history
            history['train_loss'].append(train_metrics['train_loss'])
            history['val_loss'].append(val_metrics['val_loss'])
            history['learning_rate'].append(train_metrics['learning_rate'])
            
            # Time
            epoch_time = time.time() - start_time
            
            # Logging
            print(f"Epoch {epoch}/{epochs} ({epoch_time:.2f}s):")
            print(f"  Train Loss: {train_metrics['train_loss']:.6f}")
            print(f"  Val Loss:   {val_metrics['val_loss']:.6f}")
            print(f"  LR:         {train_metrics['learning_rate']:.2e}")
            
            # Save best model
            if val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                self.save_checkpoint(save_path, metadata)
                print(f"  New best model saved!")
            
            # Early stopping
            patience = self.training_config.get('patience', 20)
            if epoch > patience:
                recent_losses = history['val_loss'][-patience:]
                if all(loss >= self.best_val_loss for loss in recent_losses):
                    print(f"Early stopping after {epoch} epochs")
                    break
        
        print(f"Training completed. Best validation loss: {self.best_val_loss:.6f}")
        
        # Load best model for final evaluation
        self.load_checkpoint(save_path)
        
        return history

def train_enhanced_diffusion_model(
    assets: List[str],
    start_date: str = "2015-01-01",
    end_date: str = "2023-12-31",
    sequence_length: int = 84,
    epochs: int = 200,
    model_config: Optional[Dict] = None,
    diffusion_config: Optional[EnhancedDiffusionConfig] = None,
    training_config: Optional[Dict] = None,
    save_path: str = "enhanced_diffusion_model.pt",
    data_dir: str = "stocks/"
) -> Tuple[str, Dict]:
    """Train the enhanced diffusion model with comprehensive setup"""
    
    set_seed(42)
    
    # Default configurations
    if model_config is None:
        model_config = {
            'base_channels': 128,
            'channel_multipliers': [1, 2, 4, 8],
            'time_dim': 512,
            'context_dim': 256,
            'condition_hidden_dim': 512,
            'dropout': 0.1,
            'attention_resolutions': [1, 2],
            'num_heads': 8,
            'use_checkpoint': False
        }
    
    if diffusion_config is None:
        diffusion_config = EnhancedDiffusionConfig(
            timesteps=1000,
            beta_schedule="cosine",
            v_prediction=True,
            classifier_free_guidance=True,
            guidance_scale=7.5,
            drop_prob=0.1
        )
    
    if training_config is None:
        training_config = {
            'batch_size': 32,
            'learning_rate': 1e-4,
            'weight_decay': 1e-6,
            'ema_decay': 0.9999,
            'ema_warmup': 2000,
            'use_amp': True,
            'use_scheduler': True,
            'scheduler_T0': 20,
            'scheduler_Tmult': 2,
            'grad_clip': 1.0,
            'patience': 30,
            'num_workers': 4
        }
    
    print("Setting up enhanced diffusion training...")
    print(f"Assets: {assets}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Sequence length: {sequence_length}")
    
    # Data preparation
    data_manager = PortfolioDataManager(assets, data_dir)
    
    # Load and process data
    merged_df, metadata = data_manager.load_and_process_data(start_date, end_date)
    print(f"Loaded {len(merged_df)} rows of data")
    print(f"Features: {len(metadata['feature_columns'])}")
    print(f"Return columns: {len(metadata['return_columns'])}")
    print(f"Condition columns: {len(metadata['condition_columns'])}")
    
    # Create datasets
    train_dataset, val_dataset, test_dataset, updated_metadata = data_manager.create_normalized_datasets(
        merged_df, metadata, sequence_length=sequence_length
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Initialize trainer
    trainer = EnhancedTrainer(model_config, diffusion_config, training_config)
    
    # Train model
    history = trainer.train(train_dataset, val_dataset, updated_metadata, epochs, save_path)
    
    print(f"Enhanced model training completed!")
    print(f"Model saved to: {save_path}")
    
    return save_path, updated_metadata

if __name__ == "__main__":
    # Example usage
    assets = ["GOOG", "NVDA", "MSFT", "AAPL", "TSLA"]
    
    model_path, metadata = train_enhanced_diffusion_model(
        assets=assets,
        start_date="2018-01-01",
        end_date="2023-12-31",
        sequence_length=84,
        epochs=100
    )