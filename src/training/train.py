"""
Main training script with comprehensive trainer class.

Implements:
1. Training loop with validation
2. Model checkpointing
3. Learning rate scheduling
4. Mixed precision training
5. Cross-validation support
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm
import os
from pathlib import Path
import json
from datetime import datetime

from .losses import MultiTaskLoss, DistillationLoss
from .distillation import KnowledgeDistillation


class Trainer:
    """
    Comprehensive trainer for nutrient analysis model.

    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration dictionary
        device: Device to use for training
        use_distillation: Whether to use knowledge distillation
        teacher_model: Teacher model for distillation (optional)
    """
    def __init__(self, model, train_loader, val_loader, config,
                 device='cuda', use_distillation=False, teacher_model=None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # Distillation
        self.use_distillation = use_distillation
        if use_distillation and teacher_model is not None:
            self.teacher = teacher_model.to(device)
            self.kd = KnowledgeDistillation(
                student_model=self.model,
                teacher_model=self.teacher,
                temperature=config.get('distillation', {}).get('temperature', 2.0),
                lambda_distill=config.get('distillation', {}).get('lambda_distill', 0.5)
            )

        # Loss function
        self.loss_fn = self._create_loss_function()

        # Optimizer
        self.optimizer = self._create_optimizer()

        # Learning rate scheduler
        self.scheduler = self._create_scheduler()

        # Mixed precision training
        self.use_amp = config.get('training', {}).get('mixed_precision', True)
        self.scaler = GradScaler() if self.use_amp else None

        # Tracking
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.train_history = []
        self.val_history = []

        # Early stopping
        self.early_stopping_patience = config.get('training', {}).get('early_stopping', {}).get('patience', 15)
        self.early_stopping_counter = 0

        # Checkpoint directory
        self.checkpoint_dir = Path(config.get('training', {}).get('checkpoint_dir', './checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _create_loss_function(self):
        """Create loss function based on configuration."""
        loss_weights = self.config.get('training', {}).get('loss_weights', {})
        micronutrient_weights = self.config.get('training', {}).get('micronutrient_weights', {})

        # Convert micronutrient weights dictionary to list
        micro_weights = [
            micronutrient_weights.get('vitamin_a', 1.2),
            micronutrient_weights.get('vitamin_c', 1.1),
            micronutrient_weights.get('vitamin_d', 1.3),
            micronutrient_weights.get('iron', 1.4),
            micronutrient_weights.get('calcium', 1.3),
            micronutrient_weights.get('zinc', 1.2)
        ]

        base_loss = MultiTaskLoss(
            alpha=loss_weights.get('alpha', 1.0),
            beta=loss_weights.get('beta', 0.5),
            gamma=loss_weights.get('gamma', 0.5),
            delta=loss_weights.get('delta', 0.3),
            micronutrient_weights=micro_weights
        )

        if self.use_distillation:
            distill_config = self.config.get('distillation', {})
            return DistillationLoss(
                temperature=distill_config.get('temperature', 2.0),
                lambda_distill=distill_config.get('lambda_distill', 0.5),
                base_loss=base_loss
            )
        else:
            return base_loss

    def _create_optimizer(self):
        """Create optimizer based on configuration."""
        train_config = self.config.get('training', {})
        optimizer_name = train_config.get('optimizer', 'Adam')
        lr = train_config.get('learning_rate', 0.001)
        weight_decay = train_config.get('weight_decay', 0.001)

        if optimizer_name == 'Adam':
            params = train_config.get('optimizer_params', {})
            return Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=(params.get('beta1', 0.9), params.get('beta2', 0.999)),
                eps=params.get('eps', 1e-8)
            )
        elif optimizer_name == 'SGD':
            return SGD(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

    def _create_scheduler(self):
        """Create learning rate scheduler."""
        scheduler_name = self.config.get('training', {}).get('scheduler', 'CosineAnnealingLR')
        scheduler_params = self.config.get('training', {}).get('scheduler_params', {})

        if scheduler_name == 'CosineAnnealingLR':
            return CosineAnnealingLR(
                self.optimizer,
                T_max=scheduler_params.get('T_max', 200),
                eta_min=scheduler_params.get('eta_min', 0.0)
            )
        elif scheduler_name == 'ReduceLROnPlateau':
            return ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=10,
                verbose=True
            )
        else:
            return None

    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")

        for batch in progress_bar:
            # Move data to device
            images = batch['image'].to(self.device)
            targets = {
                'label': batch['label'].to(self.device),
                'portion': batch['portion'].to(self.device),
                'macronutrients': batch['macronutrients'].to(self.device),
                'micronutrients': batch['micronutrients'].to(self.device)
            }

            # Forward pass with mixed precision
            with autocast(enabled=self.use_amp):
                predictions = self.model(images)

                if self.use_distillation:
                    # Get teacher predictions
                    with torch.no_grad():
                        teacher_outputs = self.teacher(images)
                    losses = self.loss_fn(predictions, teacher_outputs, targets)
                else:
                    losses = self.loss_fn(predictions, targets)

                loss = losses['total']

            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()

                # Gradient clipping
                if self.config.get('training', {}).get('grad_clip'):
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['grad_clip']
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()

                # Gradient clipping
                if self.config.get('training', {}).get('grad_clip'):
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['grad_clip']
                    )

                self.optimizer.step()

            self.optimizer.zero_grad()

            # Track loss
            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / num_batches
        return avg_loss

    @torch.no_grad()
    def validate(self):
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        num_batches = 0

        for batch in tqdm(self.val_loader, desc="Validation"):
            # Move data to device
            images = batch['image'].to(self.device)
            targets = {
                'label': batch['label'].to(self.device),
                'portion': batch['portion'].to(self.device),
                'macronutrients': batch['macronutrients'].to(self.device),
                'micronutrients': batch['micronutrients'].to(self.device)
            }

            # Forward pass
            predictions = self.model(images)

            # Compute loss
            if self.use_distillation:
                with torch.no_grad():
                    teacher_outputs = self.teacher(images)
                losses = self.loss_fn(predictions, teacher_outputs, targets)
            else:
                losses = self.loss_fn(predictions, targets)

            loss = losses['total']

            # Track metrics
            total_loss += loss.item()
            num_batches += 1

            # Compute accuracy
            _, predicted = torch.max(predictions['food_logits'], 1)
            total_correct += (predicted == targets['label']).sum().item()
            total_samples += targets['label'].size(0)

        avg_loss = total_loss / num_batches
        accuracy = total_correct / total_samples

        return avg_loss, accuracy

    def train(self, num_epochs):
        """
        Train model for specified number of epochs.

        Args:
            num_epochs: Number of epochs to train
        """
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Mixed precision: {self.use_amp}")
        print(f"Knowledge distillation: {self.use_distillation}")

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Train
            train_loss = self.train_epoch()
            self.train_history.append(train_loss)

            # Validate
            val_loss, val_acc = self.validate()
            self.val_history.append({'loss': val_loss, 'accuracy': val_acc})

            # Update learning rate scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Print epoch summary
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")

            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                self.save_checkpoint('best_model.pth', is_best=True)
                self.early_stopping_counter = 0
                print(f"New best model saved! Accuracy: {val_acc:.4f}")
            else:
                self.early_stopping_counter += 1

            # Save periodic checkpoint
            if (epoch + 1) % self.config.get('training', {}).get('save_frequency', 10) == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pth')

            # Early stopping
            if self.early_stopping_counter >= self.early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break

        print("\nTraining completed!")
        print(f"Best validation accuracy: {self.best_val_acc:.4f}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")

        # Save final model
        self.save_checkpoint('final_model.pth')

        # Save training history
        self.save_history()

    def save_checkpoint(self, filename, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.best_val_loss,
            'train_history': self.train_history,
            'val_history': self.val_history,
            'config': self.config
        }

        checkpoint_path = self.checkpoint_dir / filename
        torch.save(checkpoint, checkpoint_path)

        if is_best:
            # Also save as best model
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)

    def load_checkpoint(self, checkpoint_path):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if checkpoint.get('scheduler_state_dict') and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.train_history = checkpoint.get('train_history', [])
        self.val_history = checkpoint.get('val_history', [])

        print(f"Loaded checkpoint from epoch {self.current_epoch}")

    def save_history(self):
        """Save training history to JSON."""
        history = {
            'train_loss': self.train_history,
            'val_history': self.val_history,
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.best_val_loss,
            'timestamp': datetime.now().isoformat()
        }

        history_path = self.checkpoint_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)


if __name__ == "__main__":
    print("Trainer class created successfully!")
    print("Use scripts/train_model.py for actual training")
