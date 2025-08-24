# advanced_trainer.py - State-of-the-art training pipeline for advanced diffusion models
import os
import time
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from tqdm import tqdm
import json
import logging
from pathlib import Path

from src.advanced_diffusion import AdvancedDiffusionModel, AdvancedDiffusionConfig, AdvancedDiffusionTransformer
from src.advanced_features import AdvancedFeatureEngineer, FeatureConfig
from src.advanced_samplers import DPMSolverMultistep, HeunSampler, EDMSampler

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AdvancedTrainingConfig:
    """Configuration for advanced training pipeline"""
    # Model architecture
    hidden_dim: int = 1024
    num_layers: int = 16
    num_heads: int = 16
    max_seq_len: int = 512
    use_rotary_pos_emb: bool = True
    
    # Training hyperparameters
    batch_size: int = 32
    num_epochs: int = 200
    learning_rate: float = 1e-4
    weight_decay: float = 1e-6
    grad_clip_norm: float = 1.0
    warmup_steps: int = 10000
    
    # Optimization
    optimizer: str = "adamw"  # adamw, lion, adafactor
    scheduler: str = "cosine_with_restarts"  # cosine, cosine_with_restarts, polynomial
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    
    # Advanced training techniques
    use_mixed_precision: bool = True
    use_gradient_checkpointing: bool = True
    use_ema: bool = True
    ema_decay: float = 0.9999
    ema_warmup_steps: int = 2000
    
    # Regularization
    dropout: float = 0.1
    use_dropout_schedule: bool = True
    use_stochastic_depth: bool = True
    stochastic_depth_rate: float = 0.1
    
    # Data augmentation
    use_mixup: bool = True
    mixup_alpha: float = 0.2
    use_cutmix: bool = False
    cutmix_alpha: float = 1.0
    
    # Loss weighting and scheduling
    loss_weight_schedule: str = "constant"  # constant, linear_decay, cosine_decay
    focal_loss_alpha: float = 0.25
    focal_loss_gamma: float = 2.0
    
    # Monitoring and evaluation
    eval_every: int = 1000
    save_every: int = 5000
    num_eval_samples: int = 64
    use_wandb: bool = False
    
    # Distributed training
    use_ddp: bool = False
    local_rank: int = -1
    
    # Checkpoint management
    save_top_k: int = 3
    monitor_metric: str = "val_loss"
    early_stopping_patience: int = 20
    
    # Advanced features
    use_adversarial_training: bool = False
    adversarial_epsilon: float = 0.01
    use_spectral_normalization: bool = False

class AdvancedFinancialDataset(Dataset):
    """Advanced dataset with sophisticated augmentation and batching"""
    
    def __init__(self, data: np.ndarray, 
                 context_data: Optional[np.ndarray] = None,
                 sequence_length: int = 256,
                 prediction_horizon: int = 1,
                 stride: int = 1,
                 augment: bool = True):
        self.data = data  # (N, features)
        self.context_data = context_data  # (N, context_features)
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.stride = stride
        self.augment = augment
        
        # Calculate valid indices
        max_start = len(data) - sequence_length - prediction_horizon + 1
        self.indices = list(range(0, max_start, stride))
        
        # Precompute statistics for normalization
        self.mean = np.mean(data, axis=0, keepdims=True)
        self.std = np.std(data, axis=0, keepdims=True) + 1e-8
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        start_idx = self.indices[idx]
        end_idx = start_idx + self.sequence_length
        
        # Get sequence
        sequence = self.data[start_idx:end_idx].copy()  # (seq_len, features)
        
        # Get context if available
        if self.context_data is not None:
            context = self.context_data[end_idx - 1]  # Use latest context
        else:
            context = np.zeros(1)  # Dummy context
        
        # Data augmentation
        if self.augment and random.random() < 0.5:
            sequence = self._augment_sequence(sequence)
        
        # Normalize
        sequence = (sequence - self.mean) / self.std
        
        return {
            'sequence': torch.from_numpy(sequence).float().transpose(0, 1),  # (features, seq_len)
            'context': torch.from_numpy(context).float(),
            'timestamp': torch.tensor(start_idx, dtype=torch.long)
        }
    
    def _augment_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """Apply data augmentation techniques"""
        augmented = sequence.copy()
        
        # Gaussian noise
        if random.random() < 0.3:
            noise_std = 0.01 * np.std(sequence, axis=0, keepdims=True)
            augmented += np.random.normal(0, noise_std, sequence.shape)
        
        # Time warping (simplified)
        if random.random() < 0.2:
            # Randomly stretch/compress parts of the sequence
            warp_factor = random.uniform(0.95, 1.05)
            warped_length = int(len(sequence) * warp_factor)
            if warped_length > 10:  # Minimum length
                indices = np.linspace(0, len(sequence) - 1, warped_length).astype(int)
                augmented = sequence[indices]
                if len(augmented) != len(sequence):
                    # Pad or truncate to original length
                    if len(augmented) < len(sequence):
                        padding = len(sequence) - len(augmented)
                        augmented = np.pad(augmented, ((0, padding), (0, 0)), mode='edge')
                    else:
                        augmented = augmented[:len(sequence)]
        
        # Scaling
        if random.random() < 0.2:
            scale_factor = random.uniform(0.95, 1.05)
            augmented *= scale_factor
        
        return augmented

class AdvancedEMA:
    """Enhanced Exponential Moving Average with advanced features"""
    
    def __init__(self, model: nn.Module, decay: float = 0.9999, warmup_steps: int = 2000):
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.step = 0
        self.shadow = {}
        
        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    @torch.no_grad()
    def update(self, model: nn.Module):
        """Update EMA parameters"""
        self.step += 1
        
        # Compute effective decay with warmup
        if self.step < self.warmup_steps:
            decay = min(self.decay, self.step / self.warmup_steps)
        else:
            decay = self.decay
        
        # Update shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(decay).add_(param.data, alpha=1 - decay)
    
    @torch.no_grad()
    def apply_shadow(self, model: nn.Module):
        """Apply shadow parameters to model"""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                param.data.copy_(self.shadow[name])
    
    def state_dict(self):
        return {
            'shadow': self.shadow,
            'step': self.step,
            'decay': self.decay,
            'warmup_steps': self.warmup_steps
        }
    
    def load_state_dict(self, state_dict):
        self.shadow = state_dict['shadow']
        self.step = state_dict['step']
        self.decay = state_dict['decay']
        self.warmup_steps = state_dict['warmup_steps']

class AdvancedLRScheduler:
    """Advanced learning rate scheduling"""
    
    def __init__(self, optimizer, config: AdvancedTrainingConfig, total_steps: int):
        self.optimizer = optimizer
        self.config = config
        self.total_steps = total_steps
        self.current_step = 0
        self.base_lr = config.learning_rate
        
    def step(self):
        """Update learning rate"""
        self.current_step += 1
        
        if self.config.scheduler == "cosine":
            lr = self._cosine_schedule()
        elif self.config.scheduler == "cosine_with_restarts":
            lr = self._cosine_with_restarts_schedule()
        elif self.config.scheduler == "polynomial":
            lr = self._polynomial_schedule()
        else:
            lr = self.base_lr
        
        # Apply warmup
        if self.current_step < self.config.warmup_steps:
            warmup_factor = self.current_step / self.config.warmup_steps
            lr = lr * warmup_factor
        
        # Update optimizer
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr
    
    def _cosine_schedule(self):
        """Cosine annealing schedule"""
        progress = self.current_step / self.total_steps
        return self.base_lr * 0.5 * (1 + math.cos(math.pi * progress))
    
    def _cosine_with_restarts_schedule(self):
        """Cosine annealing with warm restarts"""
        # Simple implementation - single restart at halfway point
        if self.current_step < self.total_steps // 2:
            progress = self.current_step / (self.total_steps // 2)
        else:
            progress = (self.current_step - self.total_steps // 2) / (self.total_steps // 2)
        
        return self.base_lr * 0.5 * (1 + math.cos(math.pi * progress))
    
    def _polynomial_schedule(self):
        """Polynomial decay schedule"""
        progress = self.current_step / self.total_steps
        return self.base_lr * (1 - progress) ** 2

class AdvancedDiffusionTrainer:
    """State-of-the-art training pipeline for advanced diffusion models"""
    
    def __init__(self, config: AdvancedTrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize models
        self.model = None
        self.diffusion_model = None
        self.ema = None
        
        # Training components
        self.optimizer = None
        self.scaler = None
        self.scheduler = None
        
        # Metrics and logging
        self.step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.metrics_history = []
        
        # Setup logging
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging and monitoring"""
        if self.config.use_wandb:
            try:
                import wandb
                wandb.init(
                    project="advanced-diffusion-finance",
                    config=self.config.__dict__
                )
                self.wandb = wandb
            except ImportError:
                logger.warning("Weights & Biases not available")
                self.wandb = None
        else:
            self.wandb = None
    
    def build_model(self, input_dim: int, context_dim: Optional[int] = None):
        """Build the advanced diffusion model"""
        # Create transformer backbone
        self.model = AdvancedDiffusionTransformer(
            input_dim=input_dim,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            num_heads=self.config.num_heads,
            max_seq_len=self.config.max_seq_len,
            context_dim=context_dim,
            dropout=self.config.dropout,
            use_rotary_pos_emb=self.config.use_rotary_pos_emb
        ).to(self.device)
        
        # Create diffusion wrapper
        diffusion_config = AdvancedDiffusionConfig(
            timesteps=1000,
            beta_schedule="cosine",
            prediction_type="v",
            classifier_free_guidance=True,
            guidance_scale=7.5,
            use_karras_sigmas=True,
            min_snr_gamma=5.0,
            use_offset_noise=True,
            loss_type="huber",
            snr_weighting=True
        )
        
        self.diffusion_model = AdvancedDiffusionModel(diffusion_config, self.model)
        
        # Initialize EMA
        if self.config.use_ema:
            self.ema = AdvancedEMA(
                self.model, 
                decay=self.config.ema_decay,
                warmup_steps=self.config.ema_warmup_steps
            )
        
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        return self.model
    
    def setup_optimization(self, total_steps: int):
        """Setup optimizer and scheduler"""
        # Optimizer
        if self.config.optimizer == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=(self.config.beta1, self.config.beta2),
                eps=self.config.eps
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
        
        # Scheduler
        self.scheduler = AdvancedLRScheduler(self.optimizer, self.config, total_steps)
        
        # Mixed precision scaler
        if self.config.use_mixed_precision:
            self.scaler = GradScaler()
        
        logger.info(f"Optimization setup complete. Total steps: {total_steps}")
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step"""
        self.model.train()
        
        # Move to device
        sequence = batch['sequence'].to(self.device)  # (B, features, seq_len)
        context = batch['context'].to(self.device) if batch['context'].numel() > 1 else None
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Forward pass with mixed precision
        with autocast(enabled=self.config.use_mixed_precision):
            # Generate random timesteps
            B = sequence.shape[0]
            t = torch.randint(0, self.diffusion_model.config.timesteps, (B,), device=self.device)
            
            # Compute loss
            loss = self.diffusion_model.compute_loss(sequence, t, context)
        
        # Backward pass
        if self.config.use_mixed_precision:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
            self.optimizer.step()
        
        # Update EMA
        if self.ema is not None:
            self.ema.update(self.model)
        
        # Update scheduler
        lr = self.scheduler.step()
        
        self.step += 1
        
        return {
            'loss': loss.item(),
            'lr': lr,
            'step': self.step
        }
    
    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Evaluation on validation set"""
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(val_loader, desc="Evaluating"):
            sequence = batch['sequence'].to(self.device)
            context = batch['context'].to(self.device) if batch['context'].numel() > 1 else None
            
            # Generate random timesteps
            B = sequence.shape[0]
            t = torch.randint(0, self.diffusion_model.config.timesteps, (B,), device=self.device)
            
            # Compute loss
            loss = self.diffusion_model.compute_loss(sequence, t, context)
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return {'val_loss': avg_loss}
    
    @torch.no_grad()
    def generate_samples(self, num_samples: int = 8, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Generate samples for evaluation"""
        self.model.eval()
        
        # Use EMA model if available
        if self.ema is not None:
            original_params = {}
            for name, param in self.model.named_parameters():
                original_params[name] = param.data.clone()
            self.ema.apply_shadow(self.model)
        
        try:
            # Sample using DDIM
            shape = (num_samples, self.model.input_dim, 64)  # Shorter sequences for eval
            samples = self.diffusion_model.sample_ddim(shape, context, num_steps=50, progress=False)
            
        finally:
            # Restore original parameters
            if self.ema is not None:
                for name, param in self.model.named_parameters():
                    param.data.copy_(original_params[name])
        
        return samples
    
    def save_checkpoint(self, filepath: str, is_best: bool = False):
        """Save training checkpoint"""
        checkpoint = {
            'step': self.step,
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'best_val_loss': self.best_val_loss,
            'metrics_history': self.metrics_history
        }
        
        if self.ema is not None:
            checkpoint['ema_state_dict'] = self.ema.state_dict()
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, filepath)
        
        if is_best:
            best_path = str(Path(filepath).parent / "best_model.pt")
            torch.save(checkpoint, best_path)
        
        logger.info(f"Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load training checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.step = checkpoint['step']
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.metrics_history = checkpoint.get('metrics_history', [])
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.ema is not None and 'ema_state_dict' in checkpoint:
            self.ema.load_state_dict(checkpoint['ema_state_dict'])
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        logger.info(f"Checkpoint loaded: {filepath}")
    
    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None,
              checkpoint_dir: str = "checkpoints") -> Dict[str, List[float]]:
        """Main training loop"""
        
        # Setup
        os.makedirs(checkpoint_dir, exist_ok=True)
        total_steps = len(train_loader) * self.config.num_epochs
        self.setup_optimization(total_steps)
        
        # Training metrics
        train_losses = []
        val_losses = []
        
        logger.info(f"Starting training for {self.config.num_epochs} epochs")
        logger.info(f"Total steps: {total_steps}")
        
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            
            # Training epoch
            epoch_losses = []
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")
            
            for batch in pbar:
                metrics = self.train_step(batch)
                epoch_losses.append(metrics['loss'])
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{metrics['loss']:.6f}",
                    'lr': f"{metrics['lr']:.2e}"
                })
                
                # Evaluation
                if val_loader is not None and self.step % self.config.eval_every == 0:
                    val_metrics = self.evaluate(val_loader)
                    val_losses.append(val_metrics['val_loss'])
                    
                    # Check for best model
                    is_best = val_metrics['val_loss'] < self.best_val_loss
                    if is_best:
                        self.best_val_loss = val_metrics['val_loss']
                    
                    # Log metrics
                    log_dict = {**metrics, **val_metrics}
                    self.metrics_history.append(log_dict)
                    
                    if self.wandb:
                        self.wandb.log(log_dict, step=self.step)
                    
                    logger.info(f"Step {self.step}: Train Loss: {metrics['loss']:.6f}, "
                              f"Val Loss: {val_metrics['val_loss']:.6f}")
                
                # Save checkpoint
                if self.step % self.config.save_every == 0:
                    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{self.step}.pt")
                    self.save_checkpoint(checkpoint_path, is_best=False)
            
            # Epoch summary
            avg_train_loss = np.mean(epoch_losses)
            train_losses.append(avg_train_loss)
            
            logger.info(f"Epoch {epoch+1} completed. Avg train loss: {avg_train_loss:.6f}")
            
            # Final validation for epoch
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                val_losses.append(val_metrics['val_loss'])
                
                # Save best model
                is_best = val_metrics['val_loss'] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_metrics['val_loss']
                    checkpoint_path = os.path.join(checkpoint_dir, f"best_epoch_{epoch+1}.pt")
                    self.save_checkpoint(checkpoint_path, is_best=True)
        
        # Final checkpoint
        final_checkpoint = os.path.join(checkpoint_dir, "final_model.pt")
        self.save_checkpoint(final_checkpoint, is_best=False)
        
        logger.info(f"Training completed. Best validation loss: {self.best_val_loss:.6f}")
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'metrics_history': self.metrics_history
        }

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")