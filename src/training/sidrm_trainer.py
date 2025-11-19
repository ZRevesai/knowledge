"""
Training Pipeline for SIDRM Model.

Implements the training strategy from the paper with:
- AdamW optimizer with initial learning rate 1e-4
- Exponential decay when validation loss plateaus for 15 epochs
- Weight decay 1e-5
- Batch size 32 for population-specific training, 16 for cross-population
- Multi-objective loss function (Equation 13)

Reference: "Smart Interpretable Dietary Recommender Model for Vulnerable Populations"
by Zvinodashe Revesai and Okuthe P. Kogeda (2025)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple
import time
from pathlib import Path
import json


class SIDRMTrainer:
    """
    Trainer for SIDRM model with multi-objective optimization.

    Implements the training strategy described in Section 3.5 of the paper.
    """

    def __init__(self,
                 model: nn.Module,
                 criterion: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-5,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 patience: int = 15,
                 decay_factor: float = 0.7,
                 min_lr: float = 1e-7,
                 gradient_clip: float = 1.0,
                 save_dir: str = 'checkpoints'):
        """
        Initialize SIDRM trainer.

        Args:
            model: SIDRM model instance
            criterion: Multi-objective loss function
            train_loader: Training data loader
            val_loader: Validation data loader
            learning_rate: Initial learning rate (default: 1e-4 as per paper)
            weight_decay: Weight decay for AdamW (default: 1e-5 as per paper)
            device: Device to train on
            patience: Epochs to wait before reducing LR (default: 15 as per paper)
            decay_factor: LR decay factor (default: 0.7 as per paper)
            min_lr: Minimum learning rate
            gradient_clip: Gradient clipping threshold
            save_dir: Directory to save checkpoints
        """
        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.patience = patience
        self.decay_factor = decay_factor
        self.min_lr = min_lr
        self.gradient_clip = gradient_clip
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # AdamW optimizer as specified in paper (Section 3.5)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        # Learning rate scheduler (exponential decay on plateau)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=decay_factor,
            patience=patience,
            min_lr=min_lr,
            verbose=True
        )

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'learning_rates': [],
            'epoch_times': [],
            'loss_components': {
                'accuracy': [],
                'interpretability': [],
                'clinical': [],
                'safety': []
            }
        }

        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.epochs_without_improvement = 0

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary with training metrics
        """
        self.model.train()

        epoch_loss = 0.0
        epoch_accuracy = 0.0
        num_batches = 0

        loss_components = {
            'accuracy': 0.0,
            'interpretability': 0.0,
            'clinical': 0.0,
            'safety': 0.0
        }

        for batch_idx, batch in enumerate(self.train_loader):
            # Move data to device
            features = batch['features'].to(self.device)
            labels = batch['labels'].to(self.device)
            population_indices = batch['population'].to(self.device)

            # Forward pass
            outputs = self.model(
                features,
                population_indices=population_indices,
                return_attention=True,
                return_population_scores=False
            )

            # Compute loss
            loss_dict = self.criterion(
                outputs,
                labels,
                population_indices,
                return_components=True
            )

            loss = loss_dict['total_loss']

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip
                )

            self.optimizer.step()

            # Accumulate metrics
            epoch_loss += loss.item()

            # Compute accuracy (using deficiency probabilities)
            with torch.no_grad():
                preds = (outputs['deficiency_probabilities'] > 0.5).float()
                accuracy = (preds == labels).float().mean().item()
                epoch_accuracy += accuracy

            # Accumulate loss components
            for component in loss_components.keys():
                if f'{component}_loss' in loss_dict:
                    loss_components[component] += loss_dict[f'{component}_loss'].item()

            num_batches += 1

            # Print progress
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch {epoch} [{batch_idx + 1}/{len(self.train_loader)}] "
                      f"Loss: {loss.item():.4f}, Acc: {accuracy:.4f}")

        # Average metrics
        metrics = {
            'loss': epoch_loss / num_batches,
            'accuracy': epoch_accuracy / num_batches
        }

        # Average loss components
        for component in loss_components.keys():
            metrics[f'{component}_loss'] = loss_components[component] / num_batches

        return metrics

    def validate(self, epoch: int) -> Dict[str, float]:
        """
        Validate the model.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()

        epoch_loss = 0.0
        epoch_accuracy = 0.0
        num_batches = 0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in self.val_loader:
                # Move data to device
                features = batch['features'].to(self.device)
                labels = batch['labels'].to(self.device)
                population_indices = batch['population'].to(self.device)

                # Forward pass
                outputs = self.model(
                    features,
                    population_indices=population_indices,
                    return_attention=True,
                    return_population_scores=False
                )

                # Compute loss
                loss_dict = self.criterion(
                    outputs,
                    labels,
                    population_indices,
                    return_components=False
                )

                loss = loss_dict['total_loss']
                epoch_loss += loss.item()

                # Compute accuracy
                preds = (outputs['deficiency_probabilities'] > 0.5).float()
                accuracy = (preds == labels).float().mean().item()
                epoch_accuracy += accuracy

                # Store predictions for detailed metrics
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

                num_batches += 1

        # Average metrics
        metrics = {
            'loss': epoch_loss / num_batches,
            'accuracy': epoch_accuracy / num_batches
        }

        # Concatenate all predictions
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        # Compute per-nutrient accuracy
        per_nutrient_accuracy = (all_preds == all_labels).mean(axis=0)
        metrics['per_nutrient_accuracy'] = per_nutrient_accuracy.tolist()

        return metrics

    def train(self,
              num_epochs: int,
              early_stopping: bool = True,
              early_stopping_patience: int = 30,
              save_best: bool = True,
              verbose: bool = True) -> Dict:
        """
        Train the SIDRM model.

        Args:
            num_epochs: Number of epochs to train
            early_stopping: Whether to use early stopping
            early_stopping_patience: Patience for early stopping
            save_best: Whether to save best model
            verbose: Whether to print training progress

        Returns:
            Training history dictionary
        """
        print("=" * 70)
        print("Starting SIDRM Training")
        print("=" * 70)
        print(f"Device: {self.device}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Batch size: {self.train_loader.batch_size}")
        print(f"Number of epochs: {num_epochs}")
        print(f"Initial learning rate: {self.optimizer.param_groups[0]['lr']}")
        print(f"Weight decay: {self.optimizer.param_groups[0]['weight_decay']}")
        print("=" * 70)

        for epoch in range(1, num_epochs + 1):
            epoch_start_time = time.time()

            # Train
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate(epoch)

            # Update learning rate scheduler
            self.scheduler.step(val_metrics['loss'])

            # Record history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_accuracy'].append(train_metrics['accuracy'])
            self.history['val_accuracy'].append(val_metrics['accuracy'])
            self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])

            # Record loss components
            for component in ['accuracy', 'interpretability', 'clinical', 'safety']:
                if f'{component}_loss' in train_metrics:
                    self.history['loss_components'][component].append(
                        train_metrics[f'{component}_loss']
                    )

            epoch_time = time.time() - epoch_start_time
            self.history['epoch_times'].append(epoch_time)

            # Print epoch summary
            if verbose:
                print(f"\nEpoch {epoch}/{num_epochs}")
                print(f"Train Loss: {train_metrics['loss']:.4f}, "
                      f"Train Acc: {train_metrics['accuracy']:.4f}")
                print(f"Val Loss: {val_metrics['loss']:.4f}, "
                      f"Val Acc: {val_metrics['accuracy']:.4f}")
                print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
                print(f"Epoch Time: {epoch_time:.2f}s")

            # Check for improvement
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.best_epoch = epoch
                self.epochs_without_improvement = 0

                if save_best:
                    self.save_checkpoint(epoch, is_best=True)
                    if verbose:
                        print(f"✓ New best model saved (Val Loss: {self.best_val_loss:.4f})")
            else:
                self.epochs_without_improvement += 1

            # Early stopping
            if early_stopping and self.epochs_without_improvement >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                print(f"Best validation loss: {self.best_val_loss:.4f} at epoch {self.best_epoch}")
                break

            # Save periodic checkpoint
            if epoch % 10 == 0 and save_best:
                self.save_checkpoint(epoch, is_best=False)

        print("\n" + "=" * 70)
        print("Training completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f} at epoch {self.best_epoch}")
        print(f"Total training time: {sum(self.history['epoch_times']):.2f}s")
        print("=" * 70)

        return self.history

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        Save model checkpoint.

        Args:
            epoch: Current epoch number
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history
        }

        # Save checkpoint
        if is_best:
            checkpoint_path = self.save_dir / 'best_model.pth'
        else:
            checkpoint_path = self.save_dir / f'checkpoint_epoch_{epoch}.pth'

        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']

        print(f"Checkpoint loaded from {checkpoint_path}")
        print(f"Resuming from epoch {checkpoint['epoch']}")

    def save_history(self, filepath: str):
        """
        Save training history to JSON file.

        Args:
            filepath: Path to save history
        """
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)

        print(f"Training history saved to {filepath}")


if __name__ == "__main__":
    print("Testing SIDRM Trainer...")
    print("=" * 70)

    from ..models.sidrm import SIDRM
    from ..training.sidrm_losses import SIDRMMultiObjectiveLoss
    from ..data.nhanes_dataset import NHANESPreprocessor, create_nhanes_dataloaders

    # Create synthetic data
    preprocessor = NHANESPreprocessor(random_state=42)
    data_splits = preprocessor.generate_synthetic_nhanes_data(
        n_samples=500,  # Small for testing
        train_ratio=0.7,
        val_ratio=0.15
    )

    # Create dataloaders
    dataloaders = create_nhanes_dataloaders(
        data_splits,
        preprocessor,
        batch_size=32,
        num_workers=0
    )

    # Get dimensions
    batch = next(iter(dataloaders['train']))
    input_dim = batch['features'].shape[1]
    num_nutrients = batch['labels'].shape[1]

    print(f"\nData dimensions:")
    print(f"  Input features: {input_dim}")
    print(f"  Number of nutrients: {num_nutrients}")

    # Create model
    model = SIDRM(
        input_dim=input_dim,
        num_populations=4,
        hidden_dim=128,
        num_layers=5,
        num_attention_heads=8,
        num_nutrients=num_nutrients,
        dropout_rate=0.3
    )

    # Create criterion
    criterion = SIDRMMultiObjectiveLoss(
        num_nutrients=num_nutrients,
        alpha=0.4,
        beta=0.3,
        gamma=0.2,
        delta=0.1
    )

    # Create trainer
    trainer = SIDRMTrainer(
        model=model,
        criterion=criterion,
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        learning_rate=1e-4,
        weight_decay=1e-5,
        device='cpu',
        save_dir='test_checkpoints'
    )

    # Train for a few epochs
    print("\nTraining for 3 epochs (test)...")
    history = trainer.train(
        num_epochs=3,
        early_stopping=False,
        save_best=True,
        verbose=True
    )

    print("\nSIDRM Trainer test completed!")
