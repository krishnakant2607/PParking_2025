"""
Training pipeline for Temporal Fusion Transformer on parking occupancy data.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sys
sys.path.append('..')
from tft_model import TemporalFusionTransformer, QuantileLoss
import config
import json
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import datetime
import os

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)


class ParkingDataset(Dataset):
    """PyTorch Dataset for parking occupancy forecasting."""
    
    def __init__(self, 
                 data: pd.DataFrame,
                 encoder_length: int = 30,
                 decoder_length: int = 15,
                 historical_features: list = None,
                 future_features: list = None,
                 target_col: str = 'Occupancy_Mean',
                 scaler: Optional[StandardScaler] = None,
                 fit_scaler: bool = False):
        """
        Args:
            data: DataFrame with all features
            encoder_length: Historical context window
            decoder_length: Forecast horizon
            historical_features: List of feature names for historical context
            future_features: List of known future features (e.g., day of week)
            target_col: Name of target column
            scaler: StandardScaler for normalization
            fit_scaler: Whether to fit the scaler on this data
        """
        self.data = data.copy()
        self.encoder_length = encoder_length
        self.decoder_length = decoder_length
        self.target_col = target_col
        
        # Define feature sets if not provided
        if historical_features is None:
            # All numerical features except date and garage
            self.historical_features = [
                col for col in data.columns 
                if col not in ['Date', 'Garage'] and data[col].dtype in ['float64', 'int64']
            ]
        else:
            self.historical_features = historical_features
        
        if future_features is None:
            # Temporal features that are known in advance
            self.future_features = ['DayOfWeek', 'Month', 'Day', 'IsWeekend', 'Quarter']
        else:
            self.future_features = future_features
        
        # Remove target from historical features if present
        if self.target_col in self.historical_features:
            self.historical_features.remove(self.target_col)
        
        # Initialize or use provided scaler
        if scaler is None:
            self.scaler = StandardScaler()
            fit_scaler = True
        else:
            self.scaler = scaler
        
        # Fit scaler if needed
        if fit_scaler:
            all_features = list(set(self.historical_features + self.future_features + [self.target_col]))
            # Fill NaN before fitting scaler
            data_to_fit = self.data[all_features].ffill().bfill().fillna(0)
            self.scaler.fit(data_to_fit)
        
        # Fill NaN values before normalization
        all_features = list(set(self.historical_features + self.future_features + [self.target_col]))
        
        # Forward fill lag features, then backward fill any remaining
        self.data[all_features] = self.data[all_features].ffill().bfill().fillna(0)
        
        # Normalize data
        self.data[all_features] = self.scaler.transform(self.data[all_features])
        
        # Create sequences
        self.sequences = self._create_sequences()
        
    def _create_sequences(self) -> list:
        """Create input-output sequences for each garage."""
        sequences = []
        
        # Sort by garage and date
        df = self.data.sort_values(['Garage', 'Date']).reset_index(drop=True)
        
        # Process each garage separately
        for garage in df['Garage'].unique():
            garage_df = df[df['Garage'] == garage].reset_index(drop=True)
            
            # Skip if not enough data
            if len(garage_df) < self.encoder_length + self.decoder_length:
                continue
            
            # Create sequences with sliding window
            for i in range(len(garage_df) - self.encoder_length - self.decoder_length + 1):
                encoder_end = i + self.encoder_length
                decoder_end = encoder_end + self.decoder_length
                
                # Historical inputs (encoder)
                historical = garage_df.iloc[i:encoder_end][self.historical_features].values
                
                # Future inputs (decoder) - known future features
                future = garage_df.iloc[encoder_end:decoder_end][self.future_features].values
                
                # Target (decoder)
                target = garage_df.iloc[encoder_end:decoder_end][self.target_col].values
                
                sequences.append({
                    'historical': historical.astype(np.float32),
                    'future': future.astype(np.float32),
                    'target': target.astype(np.float32),
                    'garage': garage
                })
        
        return sequences
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq = self.sequences[idx]
        
        return {
            'historical': torch.tensor(seq['historical'], dtype=torch.float32),
            'future': torch.tensor(seq['future'], dtype=torch.float32),
            'target': torch.tensor(seq['target'], dtype=torch.float32)
        }


class TFTTrainer:
    """Trainer for Temporal Fusion Transformer."""
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 criterion: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 save_dir: str = 'models'):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in tqdm(self.train_loader, desc="Training", leave=False):
            historical = batch['historical'].to(self.device)
            future = batch['future'].to(self.device)
            target = batch['target'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions, _ = self.model(historical, future)
            
            # Loss calculation (predictions are quantiles)
            loss = self.criterion(predictions, target)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def validate(self) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating", leave=False):
                historical = batch['historical'].to(self.device)
                future = batch['future'].to(self.device)
                target = batch['target'].to(self.device)
                
                # Forward pass
                predictions, _ = self.model(historical, future)
                
                # Loss calculation
                loss = self.criterion(predictions, target)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(self, num_epochs: int, early_stopping_patience: int = 10):
        """Train the model with early stopping."""
        print(f"Training on device: {self.device}")
        print(f"Number of epochs: {num_epochs}")
        print(f"Early stopping patience: {early_stopping_patience}")
        
        patience_counter = 0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate()
            self.val_losses.append(val_loss)
            
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint('best_model.pt', epoch, val_loss)
                print(f"✓ Best model saved (val_loss: {val_loss:.4f})")
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
        
        # Save final model
        self.save_checkpoint('final_model.pt', epoch, val_loss)
        
        # Save training history
        self.save_history()
        
        print(f"\n✓ Training complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
    
    def save_checkpoint(self, filename: str, epoch: int, val_loss: float):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        
        torch.save(checkpoint, self.save_dir / filename)
    
    def save_history(self):
        """Save training history."""
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }
        
        with open(self.save_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
    
    def plot_training_history(self, save_path: str = None):
        """Plot training and validation loss curves."""
        if save_path is None:
            save_path = self.save_dir.parent / 'outputs' / 'training_history.png'
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # Linear scale plot
        ax1.plot(epochs, self.train_losses, 'b-', label='Train Loss', linewidth=2)
        ax1.plot(epochs, self.val_losses, 'r-', label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training History (Linear Scale)', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Log scale plot
        ax2.plot(epochs, self.train_losses, 'b-', label='Train Loss', linewidth=2)
        ax2.plot(epochs, self.val_losses, 'r-', label='Val Loss', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Loss (log scale)', fontsize=12)
        ax2.set_title('Training History (Log Scale)', fontsize=14, fontweight='bold')
        ax2.set_yscale('log')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Training history plot saved to: {save_path}")
    
    def evaluate_and_plot(self, test_loader: DataLoader, scaler: StandardScaler):
        """Evaluate model and generate comprehensive visualizations."""
        print("\n" + "="*60)
        print("Evaluating Model and Generating Visualizations")
        print("="*60)
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_quantiles = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                historical = batch['historical'].to(self.device)
                future = batch['future'].to(self.device)
                target = batch['target'].to(self.device)
                
                predictions, _ = self.model(historical, future)
                
                # Get median prediction (q=0.5)
                median_pred = predictions[:, :, 1]  # Middle quantile
                
                all_predictions.append(median_pred.cpu().numpy())
                all_targets.append(target.cpu().numpy())
                all_quantiles.append(predictions.cpu().numpy())
        
        # Concatenate all predictions
        predictions = np.concatenate(all_predictions, axis=0).flatten()
        targets = np.concatenate(all_targets, axis=0).flatten()
        quantiles = np.concatenate(all_quantiles, axis=0)
        
        # Calculate metrics
        mae = mean_absolute_error(targets, predictions)
        rmse = np.sqrt(mean_squared_error(targets, predictions))
        r2 = r2_score(targets, predictions)
        
        # sMAPE (Symmetric MAPE - better for normalized data)
        smape = np.mean(2.0 * np.abs(predictions - targets) / (np.abs(predictions) + np.abs(targets) + 1e-8)) * 100
        
        # Median Absolute Error (robust to outliers)
        median_ae = np.median(np.abs(predictions - targets))
        
        # Directional Accuracy (% of correct up/down predictions)
        if len(targets) > 1:
            actual_direction = np.sign(np.diff(targets))
            pred_direction = np.sign(np.diff(predictions))
            directional_accuracy = np.mean(actual_direction == pred_direction) * 100
        else:
            directional_accuracy = 0.0
        
        # Forecast Bias
        forecast_bias = np.mean(predictions - targets)
        forecast_bias_pct = (forecast_bias / (np.mean(np.abs(targets)) + 1e-8)) * 100
        
        # RMSE/MAE ratio (outlier indicator)
        rmse_mae_ratio = rmse / (mae + 1e-8)
        
        # Error percentiles
        errors = np.abs(predictions - targets)
        error_percentiles = {
            'p25': float(np.percentile(errors, 25)),
            'p50': float(np.percentile(errors, 50)),
            'p75': float(np.percentile(errors, 75)),
            'p90': float(np.percentile(errors, 90)),
            'p95': float(np.percentile(errors, 95)),
            'p99': float(np.percentile(errors, 99))
        }
        
        print(f"\n{'='*60}")
        print(f"TEST SET METRICS")
        print(f"{'='*60}")
        print(f"\n🎯 PRIMARY METRICS:")
        print(f"  R² Score:       {r2:.4f}  {'✅' if r2 > 0.7 else '⚠️' if r2 > 0.5 else '❌'}")
        print(f"  MAE:            {mae:.4f}")
        print(f"  Median AE:      {median_ae:.4f}  {'✅' if median_ae < mae else '⚠️'}")
        
        print(f"\n📊 SECONDARY METRICS:")
        print(f"  RMSE:           {rmse:.4f}")
        print(f"  RMSE/MAE Ratio: {rmse_mae_ratio:.2f}x  {'✅' if rmse_mae_ratio < 1.5 else '⚠️' if rmse_mae_ratio < 2.0 else '❌'}")
        print(f"  sMAPE:          {smape:.2f}%  {'✅' if smape < 15 else '⚠️' if smape < 25 else '❌'}")
        print(f"  Directional Acc: {directional_accuracy:.1f}%  {'✅' if directional_accuracy > 60 else '⚠️' if directional_accuracy > 50 else '❌'}")
        
        print(f"\n⚖️  BIAS ANALYSIS:")
        print(f"  Forecast Bias:  {forecast_bias:+.4f} ({forecast_bias_pct:+.2f}%)")
        print(f"  Calibration:    {'✅ Well-calibrated' if abs(forecast_bias_pct) < 5 else '⚠️ Slight bias' if abs(forecast_bias_pct) < 10 else '❌ Significant bias'}")
        
        print(f"\n📊 ERROR DISTRIBUTION:")
        print(f"  25th percentile: {error_percentiles['p25']:.4f}")
        print(f"  50th percentile: {error_percentiles['p50']:.4f}")
        print(f"  75th percentile: {error_percentiles['p75']:.4f}")
        print(f"  90th percentile: {error_percentiles['p90']:.4f}")
        print(f"  95th percentile: {error_percentiles['p95']:.4f}")
        print(f"  99th percentile: {error_percentiles['p99']:.4f}")
        print(f"  Max error:       {errors.max():.4f}")
        
        # Save metrics
        metrics = {
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'smape': float(smape),
            'median_ae': float(median_ae),
            'directional_accuracy': float(directional_accuracy),
            'forecast_bias': float(forecast_bias),
            'forecast_bias_pct': float(forecast_bias_pct),
            'rmse_mae_ratio': float(rmse_mae_ratio),
            'error_percentiles': error_percentiles,
            'max_error': float(errors.max()),
            'test_samples': len(targets)
        }
        
        outputs_dir = self.save_dir.parent / 'outputs'
        outputs_dir.mkdir(exist_ok=True)
        
        with open(outputs_dir / 'test_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"GENERATING VISUALIZATIONS")
        print(f"{'='*60}\n")
        
        # Generate all plots
        print("📊 Generating core visualizations...")
        self._plot_predictions_vs_actual(predictions, targets, outputs_dir)
        self._plot_error_analysis(predictions, targets, outputs_dir)
        self._plot_time_series_sample(predictions, targets, outputs_dir)
        self._generate_metrics_table(metrics, outputs_dir)
        
        print("\n📊 Generating advanced visualizations...")
        self._plot_residual_distribution(predictions, targets, outputs_dir)
        self._plot_error_by_magnitude(predictions, targets, outputs_dir)
        self._plot_cumulative_error(predictions, targets, outputs_dir)
        self._plot_prediction_intervals(predictions, targets, outputs_dir)
        
        print(f"\n{'='*60}")
        print(f"✅ ALL VISUALIZATIONS COMPLETE!")
        print(f"📁 Saved to: {outputs_dir}")
        print(f"{'='*60}\n")
        
        return metrics
    
    def _plot_predictions_vs_actual(self, predictions, targets, save_dir):
        """Plot predictions vs actual values with detailed analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Scatter plot
        ax = axes[0, 0]
        ax.scatter(targets, predictions, alpha=0.5, s=20)
        min_val = min(targets.min(), predictions.min())
        max_val = max(targets.max(), predictions.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        ax.set_xlabel('Actual Occupancy', fontsize=12)
        ax.set_ylabel('Predicted Occupancy', fontsize=12)
        ax.set_title('Predictions vs Actual', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Residuals plot
        ax = axes[0, 1]
        residuals = predictions - targets
        ax.scatter(predictions, residuals, alpha=0.5, s=20)
        ax.axhline(y=0, color='r', linestyle='--', lw=2)
        ax.set_xlabel('Predicted Occupancy', fontsize=12)
        ax.set_ylabel('Residuals', fontsize=12)
        ax.set_title('Residual Plot', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 3. Distribution of predictions vs actual
        ax = axes[1, 0]
        ax.hist(targets, bins=50, alpha=0.5, label='Actual', color='blue', edgecolor='black')
        ax.hist(predictions, bins=50, alpha=0.5, label='Predicted', color='red', edgecolor='black')
        ax.set_xlabel('Occupancy', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Distribution Comparison', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Error distribution
        ax = axes[1, 1]
        errors = np.abs(predictions - targets)
        ax.hist(errors, bins=50, color='orange', edgecolor='black', alpha=0.7)
        ax.axvline(x=np.mean(errors), color='r', linestyle='--', lw=2, label=f'Mean: {np.mean(errors):.2f}')
        ax.axvline(x=np.median(errors), color='g', linestyle='--', lw=2, label=f'Median: {np.median(errors):.2f}')
        ax.set_xlabel('Absolute Error', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Error Distribution', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = save_dir / 'predictions_vs_actual.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Predictions vs actual plot saved to: {save_path}")
    
    def _plot_error_analysis(self, predictions, targets, save_dir):
        """Detailed error analysis plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        errors = predictions - targets
        abs_errors = np.abs(errors)
        
        # 1. Error vs Actual Value
        ax = axes[0, 0]
        ax.scatter(targets, abs_errors, alpha=0.5, s=20, c=abs_errors, cmap='YlOrRd')
        ax.set_xlabel('Actual Occupancy', fontsize=12)
        ax.set_ylabel('Absolute Error', fontsize=12)
        ax.set_title('Error vs Actual Value', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 2. Error vs Predicted Value
        ax = axes[0, 1]
        ax.scatter(predictions, abs_errors, alpha=0.5, s=20, c=abs_errors, cmap='YlOrRd')
        ax.set_xlabel('Predicted Occupancy', fontsize=12)
        ax.set_ylabel('Absolute Error', fontsize=12)
        ax.set_title('Error vs Predicted Value', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 3. Relative Error Distribution
        ax = axes[1, 0]
        mask = targets != 0
        relative_errors = (abs_errors[mask] / np.abs(targets[mask])) * 100
        relative_errors = np.clip(relative_errors, 0, 200)  # Clip extreme values
        ax.hist(relative_errors, bins=50, color='purple', edgecolor='black', alpha=0.7)
        ax.axvline(x=np.median(relative_errors), color='r', linestyle='--', lw=2, 
                   label=f'Median: {np.median(relative_errors):.1f}%')
        ax.set_xlabel('Relative Error (%)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Relative Error Distribution', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Error Percentiles
        ax = axes[1, 1]
        percentiles = [50, 75, 90, 95, 99]
        values = [np.percentile(abs_errors, p) for p in percentiles]
        colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.9, len(percentiles)))
        bars = ax.bar(range(len(percentiles)), values, color=colors, edgecolor='black', linewidth=1.5)
        ax.set_xticks(range(len(percentiles)))
        ax.set_xticklabels([f'{p}th' for p in percentiles])
        ax.set_xlabel('Percentile', fontsize=12)
        ax.set_ylabel('Absolute Error', fontsize=12)
        ax.set_title('Error Percentiles', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        save_path = save_dir / 'error_analysis.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Error analysis plot saved to: {save_path}")
    
    def _plot_time_series_sample(self, predictions, targets, save_dir, n_points=500):
        """Plot time series comparison for a sample of predictions."""
        # Take a sample of consecutive points
        if len(predictions) > n_points:
            start_idx = len(predictions) // 2 - n_points // 2
            end_idx = start_idx + n_points
            sample_pred = predictions[start_idx:end_idx]
            sample_target = targets[start_idx:end_idx]
        else:
            sample_pred = predictions
            sample_target = targets
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Full sample
        ax1.plot(sample_target, 'b-', label='Actual', linewidth=1.5, alpha=0.7)
        ax1.plot(sample_pred, 'r-', label='Predicted', linewidth=1.5, alpha=0.7)
        ax1.set_xlabel('Time Step', fontsize=12)
        ax1.set_ylabel('Occupancy', fontsize=12)
        ax1.set_title('Time Series Comparison (Sample)', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Zoomed view (first 100 points or less)
        zoom_points = min(100, len(sample_pred))
        ax2.plot(sample_target[:zoom_points], 'b-', label='Actual', linewidth=2, alpha=0.7, marker='o', markersize=4)
        ax2.plot(sample_pred[:zoom_points], 'r-', label='Predicted', linewidth=2, alpha=0.7, marker='s', markersize=4)
        ax2.set_xlabel('Time Step', fontsize=12)
        ax2.set_ylabel('Occupancy', fontsize=12)
        ax2.set_title(f'Time Series Comparison (Zoomed - First {zoom_points} Points)', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = save_dir / 'time_series_sample.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Time series comparison plot saved to: {save_path}")
    
    def _generate_metrics_table(self, metrics, save_dir):
        """Generate and save a comprehensive metrics summary table."""
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare table data with status indicators
        r2_status = '✅' if metrics['r2'] > 0.7 else '⚠️' if metrics['r2'] > 0.5 else '❌'
        smape_status = '✅' if metrics['smape'] < 15 else '⚠️' if metrics['smape'] < 25 else '❌'
        da_status = '✅' if metrics['directional_accuracy'] > 60 else '⚠️' if metrics['directional_accuracy'] > 50 else '❌'
        bias_status = '✅' if abs(metrics['forecast_bias_pct']) < 5 else '⚠️' if abs(metrics['forecast_bias_pct']) < 10 else '❌'
        
        table_data = [
            ['Metric', 'Value', 'Status', 'Description'],
            ['', '', '', ''],  # Spacer
            ['PRIMARY METRICS', '', '', ''],
            ['R² Score', f"{metrics['r2']:.4f}", r2_status, 'Variance explained by model'],
            ['MAE', f"{metrics['mae']:.4f}", '', 'Mean Absolute Error'],
            ['Median AE', f"{metrics['median_ae']:.4f}", '', 'Median Absolute Error (robust)'],
            ['', '', '', ''],  # Spacer
            ['SECONDARY METRICS', '', '', ''],
            ['RMSE', f"{metrics['rmse']:.4f}", '', 'Root Mean Squared Error'],
            ['RMSE/MAE', f"{metrics['rmse_mae_ratio']:.2f}x", '', 'Outlier indicator (<1.5 ideal)'],
            ['sMAPE', f"{metrics['smape']:.2f}%", smape_status, 'Symmetric MAPE (0-200%)'],
            ['Directional Acc', f"{metrics['directional_accuracy']:.1f}%", da_status, '% correct up/down predictions'],
            ['', '', '', ''],  # Spacer
            ['BIAS ANALYSIS', '', '', ''],
            ['Forecast Bias', f"{metrics['forecast_bias']:+.4f}", '', 'Mean(pred - actual)'],
            ['Bias %', f"{metrics['forecast_bias_pct']:+.2f}%", bias_status, 'Relative to mean'],
            ['', '', '', ''],  # Spacer
            ['ERROR DISTRIBUTION', '', '', ''],
            ['25th Percentile', f"{metrics['error_percentiles']['p25']:.4f}", '', '25% of errors below this'],
            ['50th Percentile', f"{metrics['error_percentiles']['p50']:.4f}", '', 'Median error'],
            ['75th Percentile', f"{metrics['error_percentiles']['p75']:.4f}", '', '75% of errors below this'],
            ['90th Percentile', f"{metrics['error_percentiles']['p90']:.4f}", '', '90% of errors below this'],
            ['95th Percentile', f"{metrics['error_percentiles']['p95']:.4f}", '', '95% of errors below this'],
            ['99th Percentile', f"{metrics['error_percentiles']['p99']:.4f}", '', '99% of errors below this'],
            ['Max Error', f"{metrics['max_error']:.4f}", '', 'Largest prediction error'],
            ['', '', '', ''],  # Spacer
            ['DATASET INFO', '', '', ''],
            ['Test Samples', f"{metrics['test_samples']:,}", '', 'Number of test samples']
        ]
        
        # Create table
        table = ax.table(cellText=table_data, cellLoc='left', loc='center',
                        colWidths=[0.25, 0.20, 0.10, 0.45])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.8)
        
        # Style header row
        for i in range(4):
            cell = table[(0, i)]
            cell.set_facecolor('#1f77b4')
            cell.set_text_props(weight='bold', color='white', size=11)
        
        # Style section headers
        section_rows = [2, 7, 13, 17, 26]
        for row in section_rows:
            for j in range(4):
                cell = table[(row, j)]
                cell.set_facecolor('#4CAF50')
                cell.set_text_props(weight='bold', color='white')
        
        # Style spacer rows
        spacer_rows = [1, 6, 12, 16, 25]
        for row in spacer_rows:
            for j in range(4):
                cell = table[(row, j)]
                cell.set_facecolor('#e0e0e0')
                cell.set_height(0.5)
        
        # Style data rows with alternating colors
        for i in range(len(table_data)):
            if i not in [0] + section_rows + spacer_rows:
                for j in range(4):
                    cell = table[(i, j)]
                    if i % 2 == 0:
                        cell.set_facecolor('#f8f9fa')
                    else:
                        cell.set_facecolor('white')
        
        # Add title
        plt.title('📊 Comprehensive Model Performance Metrics', 
                 fontsize=16, fontweight='bold', pad=20)
        
        save_path = save_dir / 'metrics_summary.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Comprehensive metrics table saved to: {save_path}")
    
    def _plot_residual_distribution(self, predictions, targets, save_dir):
        """Plot detailed residual distribution analysis."""
        residuals = predictions - targets
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Histogram of residuals
        ax = axes[0, 0]
        ax.hist(residuals, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        ax.axvline(np.mean(residuals), color='green', linestyle='--', linewidth=2, 
                  label=f'Mean: {np.mean(residuals):.4f}')
        ax.set_xlabel('Residual (Predicted - Actual)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Residual Distribution', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Q-Q plot for normality check
        ax = axes[0, 1]
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title('Q-Q Plot (Normality Test)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 3. Residuals vs fitted values (heteroscedasticity check)
        ax = axes[1, 0]
        ax.scatter(predictions, residuals, alpha=0.5, s=20)
        ax.axhline(0, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Predicted Values', fontsize=12)
        ax.set_ylabel('Residuals', fontsize=12)
        ax.set_title('Residuals vs Fitted Values', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 4. Absolute residuals vs fitted (scale-location plot)
        ax = axes[1, 1]
        ax.scatter(predictions, np.abs(residuals), alpha=0.5, s=20, color='orange')
        ax.set_xlabel('Predicted Values', fontsize=12)
        ax.set_ylabel('|Residuals|', fontsize=12)
        ax.set_title('Scale-Location Plot', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = save_dir / 'residual_distribution.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Residual distribution plot saved to: {save_path}")
    
    def _plot_error_by_magnitude(self, predictions, targets, save_dir):
        """Plot how error varies with prediction magnitude."""
        errors = np.abs(predictions - targets)
        
        # Create bins for prediction magnitude
        n_bins = 10
        pred_bins = np.linspace(predictions.min(), predictions.max(), n_bins + 1)
        bin_centers = (pred_bins[:-1] + pred_bins[1:]) / 2
        
        mean_errors = []
        std_errors = []
        
        for i in range(n_bins):
            mask = (predictions >= pred_bins[i]) & (predictions < pred_bins[i+1])
            if mask.sum() > 0:
                mean_errors.append(errors[mask].mean())
                std_errors.append(errors[mask].std())
            else:
                mean_errors.append(0)
                std_errors.append(0)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.errorbar(bin_centers, mean_errors, yerr=std_errors, 
                   marker='o', markersize=8, linewidth=2, capsize=5,
                   label='Mean ± Std')
        ax.set_xlabel('Prediction Magnitude', fontsize=12)
        ax.set_ylabel('Absolute Error', fontsize=12)
        ax.set_title('Error vs Prediction Magnitude', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = save_dir / 'error_by_magnitude.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Error by magnitude plot saved to: {save_path}")
    
    def _plot_cumulative_error(self, predictions, targets, save_dir):
        """Plot cumulative distribution of errors."""
        errors = np.abs(predictions - targets)
        sorted_errors = np.sort(errors)
        cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(sorted_errors, cumulative, linewidth=2, color='blue')
        
        # Add reference lines for key percentiles
        percentiles = [50, 75, 90, 95, 99]
        for p in percentiles:
            val = np.percentile(errors, p)
            ax.axvline(val, color='red', linestyle='--', alpha=0.5)
            ax.text(val, p, f'  {p}th: {val:.4f}', fontsize=10, va='center')
        
        ax.set_xlabel('Absolute Error', fontsize=12)
        ax.set_ylabel('Cumulative Percentage (%)', fontsize=12)
        ax.set_title('Cumulative Error Distribution', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = save_dir / 'cumulative_error.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Cumulative error plot saved to: {save_path}")
    
    def _plot_prediction_intervals(self, predictions, targets, save_dir, sample_size=200):
        """Plot prediction intervals confidence."""
        # Take a sample
        if len(predictions) > sample_size:
            indices = np.random.choice(len(predictions), sample_size, replace=False)
            indices = np.sort(indices)
            sample_pred = predictions[indices]
            sample_target = targets[indices]
        else:
            sample_pred = predictions
            sample_target = targets
        
        # Calculate prediction intervals (using residual std as approximation)
        residuals = predictions - targets
        residual_std = np.std(residuals)
        
        lower_bound = sample_pred - 1.96 * residual_std  # 95% CI
        upper_bound = sample_pred + 1.96 * residual_std
        
        fig, ax = plt.subplots(figsize=(15, 6))
        
        x = np.arange(len(sample_pred))
        ax.fill_between(x, lower_bound, upper_bound, alpha=0.3, label='95% Prediction Interval')
        ax.plot(x, sample_pred, 'b-', linewidth=2, label='Predicted', alpha=0.7)
        ax.plot(x, sample_target, 'r.', markersize=6, label='Actual', alpha=0.7)
        
        ax.set_xlabel('Sample Index', fontsize=12)
        ax.set_ylabel('Occupancy', fontsize=12)
        ax.set_title('Predictions with 95% Confidence Intervals', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = save_dir / 'prediction_intervals.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Prediction intervals plot saved to: {save_path}")


def prepare_data(data_path: str = 'data/processed_data.csv',
                encoder_length: int = 30,
                decoder_length: int = 15,
                train_ratio: float = 0.7,
                val_ratio: float = 0.15,
                batch_size: int = 32) -> Tuple[DataLoader, DataLoader, DataLoader, StandardScaler]:
    """
    Prepare data loaders for training.
    
    Returns:
        train_loader, val_loader, test_loader, scaler
    """
    # Load data
    print("Loading data...")
    df = pd.read_csv(data_path, parse_dates=['Date'])
    
    print(f"Total records: {len(df)}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"Garages: {df['Garage'].nunique()}")
    
    # Sort by garage and date
    df = df.sort_values(['Garage', 'Date']).reset_index(drop=True)
    
    # Split data by time (not random) for each garage
    train_dfs = []
    val_dfs = []
    test_dfs = []
    
    for garage in df['Garage'].unique():
        garage_df = df[df['Garage'] == garage].reset_index(drop=True)
        n = len(garage_df)
        
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_dfs.append(garage_df.iloc[:train_end])
        val_dfs.append(garage_df.iloc[train_end:val_end])
        test_dfs.append(garage_df.iloc[val_end:])
    
    train_df = pd.concat(train_dfs, ignore_index=True)
    val_df = pd.concat(val_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)
    
    print(f"\nData splits:")
    print(f"  Train: {len(train_df)} records")
    print(f"  Val:   {len(val_df)} records")
    print(f"  Test:  {len(test_df)} records")
    
    # Create datasets (fit scaler only on training data)
    train_dataset = ParkingDataset(
        train_df, encoder_length, decoder_length, 
        fit_scaler=True
    )
    
    val_dataset = ParkingDataset(
        val_df, encoder_length, decoder_length,
        scaler=train_dataset.scaler, fit_scaler=False
    )
    
    test_dataset = ParkingDataset(
        test_df, encoder_length, decoder_length,
        scaler=train_dataset.scaler, fit_scaler=False
    )
    
    print(f"\nSequences created:")
    print(f"  Train: {len(train_dataset)} sequences")
    print(f"  Val:   {len(val_dataset)} sequences")
    print(f"  Test:  {len(test_dataset)} sequences")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    
    # Save scaler and feature info
    save_dir = Path('models')
    save_dir.mkdir(exist_ok=True)
    
    with open(save_dir / 'scaler.pkl', 'wb') as f:
        pickle.dump(train_dataset.scaler, f)
    
    feature_info = {
        'historical_features': train_dataset.historical_features,
        'future_features': train_dataset.future_features,
        'target_col': train_dataset.target_col,
        'encoder_length': encoder_length,
        'decoder_length': decoder_length
    }
    
    with open(save_dir / 'feature_info.json', 'w') as f:
        json.dump(feature_info, f, indent=2)
    
    print("\n✓ Scaler and feature info saved to models/")
    
    return train_loader, val_loader, test_loader, train_dataset.scaler


def main():
    """Main training script."""
    # Start timing
    start_time = time.time()
    
    # Hyperparameters
    ENCODER_LENGTH = 30  # 30 days of history
    DECODER_LENGTH = 15  # Predict next 15 days
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100
    HIDDEN_DIM = 128  # Daily model uses 128
    NUM_HEADS = 4
    DROPOUT = 0.1
    QUANTILES = [0.1, 0.5, 0.9]
    
    # GPU selection - Use GPU 1 if specified
    USE_GPU_1 = os.environ.get('CUDA_VISIBLE_DEVICES') == '1'
    
    print("=" * 60)
    print("Daily TFT Model - Full Dataset Training")
    print("=" * 60)
    print(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"GPU Selection: {'GPU 1 (cuda:1)' if USE_GPU_1 else 'GPU 0 or CPU'}")
    
    # Prepare data
    train_loader, val_loader, test_loader, scaler = prepare_data(
        data_path='data/processed_data.csv',
        encoder_length=ENCODER_LENGTH,
        decoder_length=DECODER_LENGTH,
        batch_size=BATCH_SIZE
    )
    
    # Get feature dimensions from first batch
    sample_batch = next(iter(train_loader))
    historical_features = sample_batch['historical'].shape[2]
    future_features = sample_batch['future'].shape[2]
    
    print(f"\nFeature dimensions:")
    print(f"  Historical features: {historical_features}")
    print(f"  Future features: {future_features}")
    
    # Initialize model
    print("\nInitializing model...")
    model = TemporalFusionTransformer(
        static_features=0,
        historical_features=historical_features,
        future_features=future_features,
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        num_encoder_steps=ENCODER_LENGTH,
        num_decoder_steps=DECODER_LENGTH,
        dropout=DROPOUT,
        num_quantiles=len(QUANTILES)
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Device selection
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"\n✓ Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = 'cpu'
        print("\n⚠ No GPU available, using CPU")
    
    # Loss and optimizer
    criterion = QuantileLoss(quantiles=QUANTILES)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )
    
    # Initialize trainer
    trainer = TFTTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        save_dir='models'
    )
    
    # Train
    print("\n" + "=" * 60)
    print("Starting Training...")
    print("=" * 60)
    
    trainer.train(num_epochs=NUM_EPOCHS, early_stopping_patience=10)
    
    # Training time
    training_time = time.time() - start_time
    print(f"\n⏱️  Total training time: {training_time/60:.2f} minutes")
    
    # Plot training history
    print("\n" + "=" * 60)
    print("Generating Visualizations...")
    print("=" * 60)
    
    trainer.plot_training_history()
    
    # Load best model for evaluation
    print("\nLoading best model for evaluation...")
    best_checkpoint = torch.load('models/best_model.pt', map_location=device)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    trainer.model = model.to(device)
    
    # Evaluate and generate plots
    metrics = trainer.evaluate_and_plot(test_loader, scaler)
    
    # Save final summary
    summary = {
        'training_date': datetime.now().isoformat(),
        'training_time_minutes': training_time / 60,
        'total_epochs': len(trainer.train_losses),
        'best_val_loss': trainer.best_val_loss,
        'test_metrics': metrics,
        'model_config': {
            'hidden_dim': HIDDEN_DIM,
            'num_heads': NUM_HEADS,
            'encoder_length': ENCODER_LENGTH,
            'decoder_length': DECODER_LENGTH,
            'dropout': DROPOUT
        },
        'device': device,
        'total_parameters': total_params
    }
    
    with open('outputs/training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nBest Model Performance:")
    print(f"  R² Score:  {metrics['r2']:.4f}")
    print(f"  MAE:       {metrics['mae']:.4f}")
    print(f"  RMSE:      {metrics['rmse']:.4f}")
    print(f"  sMAPE:     {metrics['smape']:.2f}%")
    print(f"  Dir. Acc:  {metrics['directional_accuracy']:.1f}%")
    print(f"\nAll results saved to:")
    print(f"  Models:    {Path('models').absolute()}")
    print(f"  Outputs:   {Path('outputs').absolute()}")
    print(f"  Summary:   outputs/training_summary.json")
    
    return metrics


if __name__ == "__main__":
    main()
