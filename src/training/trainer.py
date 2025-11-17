"""
Trainer class for KGNN model.

Handles training loop, validation, checkpointing, and early stopping.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Optional, List
from tqdm import tqdm
import os
import json
from pathlib import Path

from .losses import MultiObjectiveLoss


class KGNNTrainer:
    """
    Trainer for Knowledge-Guided Neural Networks.

    Manages the complete training pipeline including:
    - Training and validation loops
    - Loss computation and optimization
    - Learning rate scheduling
    - Early stopping
    - Model checkpointing
    - Training history tracking
    """

    def __init__(self,
                 model: nn.Module,
                 criterion: MultiObjectiveLoss,
                 optimizer: optim.Optimizer,
                 device: str = 'cpu',
                 scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
                 early_stopping_patience: int = 15,
                 gradient_clip: float = 1.0,
                 checkpoint_dir: str = './checkpoints',
                 log_interval: int = 10):
        """
        Initialize trainer.

        Args:
            model: KGNN model to train
            criterion: Multi-objective loss function
            optimizer: Optimizer for training
            device: Device for training ('cpu' or 'cuda')
            scheduler: Optional learning rate scheduler
            early_stopping_patience: Patience for early stopping
            gradient_clip: Maximum gradient norm for clipping
            checkpoint_dir: Directory for saving checkpoints
            log_interval: Interval for logging training progress
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.early_stopping_patience = early_stopping_patience
        self.gradient_clip = gradient_clip
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_interval = log_interval

        # Move model to device
        self.model.to(self.device)

        # Training history
        self.history = {
            'train_loss': [],
            'train_pred_loss': [],
            'train_know_loss': [],
            'train_expl_loss': [],
            'val_loss': [],
            'val_pred_loss': [],
            'val_know_loss': [],
            'val_expl_loss': [],
            'learning_rates': []
        }

        # Early stopping tracking
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.best_model_state = None

        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader: DataLoader for training data

        Returns:
            Dictionary with average losses for the epoch
        """
        self.model.train()

        epoch_losses = {
            'total_loss': 0.0,
            'prediction_loss': 0.0,
            'knowledge_loss': 0.0,
            'explanation_loss': 0.0
        }

        num_batches = 0

        for batch_idx, (features, labels) in enumerate(tqdm(train_loader, desc="Training")):
            # Move data to device
            features = features.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            outputs = self.model(features, return_attention=True)
            predictions = outputs['predictions']
            attention_weights = outputs.get('attention_weights', None)

            # Compute loss
            loss_dict = self.criterion(
                model=self.model,
                predictions=predictions,
                targets=labels,
                attention_weights=attention_weights
            )

            total_loss = loss_dict['total_loss']

            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()

            # Gradient clipping
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip
                )

            # Optimizer step
            self.optimizer.step()

            # Accumulate losses
            epoch_losses['total_loss'] += total_loss.item()
            epoch_losses['prediction_loss'] += loss_dict['prediction_loss'].item()
            epoch_losses['knowledge_loss'] += loss_dict['knowledge_loss'].item()
            epoch_losses['explanation_loss'] += loss_dict['explanation_loss'].item()

            num_batches += 1

            # Logging
            if batch_idx % self.log_interval == 0:
                print(f"Batch {batch_idx}/{len(train_loader)}, "
                      f"Loss: {total_loss.item():.4f}")

        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches

        return epoch_losses

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate the model.

        Args:
            val_loader: DataLoader for validation data

        Returns:
            Dictionary with average validation losses
        """
        self.model.eval()

        epoch_losses = {
            'total_loss': 0.0,
            'prediction_loss': 0.0,
            'knowledge_loss': 0.0,
            'explanation_loss': 0.0
        }

        num_batches = 0

        with torch.no_grad():
            for features, labels in tqdm(val_loader, desc="Validation"):
                # Move data to device
                features = features.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self.model(features, return_attention=True)
                predictions = outputs['predictions']
                attention_weights = outputs.get('attention_weights', None)

                # Compute loss
                loss_dict = self.criterion(
                    model=self.model,
                    predictions=predictions,
                    targets=labels,
                    attention_weights=attention_weights
                )

                # Accumulate losses
                epoch_losses['total_loss'] += loss_dict['total_loss'].item()
                epoch_losses['prediction_loss'] += loss_dict['prediction_loss'].item()
                epoch_losses['knowledge_loss'] += loss_dict['knowledge_loss'].item()
                epoch_losses['explanation_loss'] += loss_dict['explanation_loss'].item()

                num_batches += 1

        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches

        return epoch_losses

    def train(self,
              train_loader: DataLoader,
              val_loader: DataLoader,
              num_epochs: int = 100) -> Dict:
        """
        Train the model for multiple epochs.

        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            num_epochs: Number of epochs to train

        Returns:
            Training history dictionary
        """
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 50)

            # Train
            train_losses = self.train_epoch(train_loader)

            # Validate
            val_losses = self.validate(val_loader)

            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_losses['total_loss'])
                else:
                    self.scheduler.step()

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']

            # Update history
            self.history['train_loss'].append(train_losses['total_loss'])
            self.history['train_pred_loss'].append(train_losses['prediction_loss'])
            self.history['train_know_loss'].append(train_losses['knowledge_loss'])
            self.history['train_expl_loss'].append(train_losses['explanation_loss'])
            self.history['val_loss'].append(val_losses['total_loss'])
            self.history['val_pred_loss'].append(val_losses['prediction_loss'])
            self.history['val_know_loss'].append(val_losses['knowledge_loss'])
            self.history['val_expl_loss'].append(val_losses['explanation_loss'])
            self.history['learning_rates'].append(current_lr)

            # Print epoch summary
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"  Train Loss: {train_losses['total_loss']:.4f} "
                  f"(Pred: {train_losses['prediction_loss']:.4f}, "
                  f"Know: {train_losses['knowledge_loss']:.4f}, "
                  f"Expl: {train_losses['explanation_loss']:.4f})")
            print(f"  Val Loss: {val_losses['total_loss']:.4f} "
                  f"(Pred: {val_losses['prediction_loss']:.4f}, "
                  f"Know: {val_losses['knowledge_loss']:.4f}, "
                  f"Expl: {val_losses['explanation_loss']:.4f})")
            print(f"  Learning Rate: {current_lr:.6f}")

            # Early stopping check
            if val_losses['total_loss'] < self.best_val_loss:
                self.best_val_loss = val_losses['total_loss']
                self.epochs_without_improvement = 0
                self.best_model_state = self.model.state_dict().copy()

                # Save best model
                self.save_checkpoint(
                    epoch=epoch,
                    is_best=True,
                    val_loss=val_losses['total_loss']
                )

                print(f"  ✓ New best model! Val loss: {self.best_val_loss:.4f}")
            else:
                self.epochs_without_improvement += 1
                print(f"  No improvement for {self.epochs_without_improvement} epochs")

                if self.epochs_without_improvement >= self.early_stopping_patience:
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                    break

            # Save checkpoint periodically
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(
                    epoch=epoch,
                    is_best=False,
                    val_loss=val_losses['total_loss']
                )

        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"\nLoaded best model with validation loss: {self.best_val_loss:.4f}")

        # Save final training history
        self.save_history()

        print("\nTraining completed!")
        return self.history

    def save_checkpoint(self,
                       epoch: int,
                       is_best: bool = False,
                       val_loss: float = 0.0):
        """
        Save model checkpoint.

        Args:
            epoch: Current epoch number
            is_best: Whether this is the best model so far
            val_loss: Validation loss
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'history': self.history
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        # Save checkpoint
        if is_best:
            path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, path)
            print(f"  Saved best model to {path}")
        else:
            path = self.checkpoint_dir / f'checkpoint_epoch_{epoch + 1}.pt'
            torch.save(checkpoint, path)
            print(f"  Saved checkpoint to {path}")

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.history = checkpoint.get('history', self.history)

        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"  Epoch: {checkpoint['epoch']}")
        print(f"  Val Loss: {checkpoint['val_loss']:.4f}")

    def save_history(self):
        """Save training history to JSON file."""
        history_path = self.checkpoint_dir / 'training_history.json'

        # Convert numpy arrays to lists for JSON serialization
        history_serializable = {}
        for key, value in self.history.items():
            if isinstance(value, list):
                history_serializable[key] = value
            else:
                history_serializable[key] = float(value)

        with open(history_path, 'w') as f:
            json.dump(history_serializable, f, indent=2)

        print(f"Saved training history to {history_path}")

    def get_history(self) -> Dict:
        """
        Get training history.

        Returns:
            Dictionary with training history
        """
        return self.history


if __name__ == "__main__":
    print("Testing KGNN Trainer...")

    # This is a minimal test - actual usage would involve real data and model
    from torch.utils.data import TensorDataset

    # Create dummy data
    X_train = torch.randn(200, 105)
    y_train = torch.randint(0, 2, (200, 12)).float()
    X_val = torch.randn(50, 105)
    y_val = torch.randint(0, 2, (50, 12)).float()

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Create dummy model
    class DummyKGNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(105, 64),
                nn.ReLU(),
                nn.Linear(64, 12),
                nn.Sigmoid()
            )

        def forward(self, x, return_attention=False):
            pred = self.fc(x)
            output = {'predictions': pred}
            if return_attention:
                output['attention_weights'] = torch.rand(x.size(0), 64)
            return output

    model = DummyKGNN()
    criterion = MultiObjectiveLoss(lambda_k=0.0, lambda_e=0.0)  # No knowledge/explanation loss for dummy test
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    trainer = KGNNTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device='cpu',
        early_stopping_patience=5
    )

    print("Starting dummy training...")
    history = trainer.train(train_loader, val_loader, num_epochs=3)

    print("\nTraining history:")
    print(f"  Train losses: {history['train_loss']}")
    print(f"  Val losses: {history['val_loss']}")

    print("\nTrainer test completed!")
