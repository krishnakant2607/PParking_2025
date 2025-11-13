"""
Evaluation and visualization tools for TFT predictions.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple, List
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json


class ForecastEvaluator:
    """Evaluate forecasting performance."""
    
    def __init__(self):
        self.metrics = {}
        
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                         prefix: str = "") -> Dict[str, float]:
        """
        Calculate forecasting metrics.
        
        Args:
            y_true: Ground truth values [n_samples, horizon]
            y_pred: Predicted values [n_samples, horizon]
            prefix: Prefix for metric names
            
        Returns:
            Dictionary of metrics
        """
        # Flatten for overall metrics
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        # Remove NaN values
        mask = ~(np.isnan(y_true_flat) | np.isnan(y_pred_flat))
        y_true_flat = y_true_flat[mask]
        y_pred_flat = y_pred_flat[mask]
        
        metrics = {}
        
        # MAE
        metrics[f'{prefix}MAE'] = mean_absolute_error(y_true_flat, y_pred_flat)
        
        # RMSE
        metrics[f'{prefix}RMSE'] = np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))
        
        # MAPE
        mape = np.mean(np.abs((y_true_flat - y_pred_flat) / (y_true_flat + 1e-8))) * 100
        metrics[f'{prefix}MAPE'] = mape
        
        # R2
        metrics[f'{prefix}R2'] = r2_score(y_true_flat, y_pred_flat)
        
        # Per-horizon metrics
        for h in range(y_true.shape[1]):
            y_true_h = y_true[:, h]
            y_pred_h = y_pred[:, h]
            
            # Remove NaN
            mask_h = ~(np.isnan(y_true_h) | np.isnan(y_pred_h))
            y_true_h = y_true_h[mask_h]
            y_pred_h = y_pred_h[mask_h]
            
            if len(y_true_h) > 0:
                metrics[f'{prefix}MAE_Day{h+1}'] = mean_absolute_error(y_true_h, y_pred_h)
                metrics[f'{prefix}RMSE_Day{h+1}'] = np.sqrt(mean_squared_error(y_true_h, y_pred_h))
        
        return metrics
    
    @staticmethod
    def calculate_quantile_metrics(y_true: np.ndarray, 
                                   quantile_preds: Dict[float, np.ndarray]) -> Dict[str, float]:
        """
        Calculate quantile-specific metrics.
        
        Args:
            y_true: Ground truth [n_samples, horizon]
            quantile_preds: Dictionary {quantile: predictions}
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        for q, y_pred in quantile_preds.items():
            # Pinball loss
            errors = y_true - y_pred
            pinball = np.where(errors >= 0, q * errors, (q - 1) * errors)
            metrics[f'Pinball_Q{int(q*100)}'] = np.mean(pinball)
            
            # Coverage (for prediction intervals)
            if q == 0.5:
                # Check if true value is within prediction interval
                if 0.1 in quantile_preds and 0.9 in quantile_preds:
                    lower = quantile_preds[0.1]
                    upper = quantile_preds[0.9]
                    coverage = np.mean((y_true >= lower) & (y_true <= upper))
                    metrics['Coverage_80'] = coverage
        
        return metrics
    
    def evaluate_model(self, model, data_loader, device: str = 'cpu',
                      scaler=None) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
        """
        Evaluate model on a dataset.
        
        Returns:
            metrics, predictions, targets
        """
        model.eval()
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in data_loader:
                historical = batch['historical'].to(device)
                future = batch['future'].to(device)
                target = batch['target'].to(device)
                
                # Forward pass
                predictions, _ = model(historical, future)
                
                # Take median (quantile 0.5, middle index)
                median_pred = predictions[:, :, 1]  # Assuming [0.1, 0.5, 0.9]
                
                all_predictions.append(median_pred.cpu().numpy())
                all_targets.append(target.cpu().numpy())
        
        # Concatenate
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        # Inverse transform if scaler provided
        if scaler is not None:
            # Note: This assumes target is the first feature in scaler
            # You may need to adjust based on your scaler setup
            pass  # Implement inverse transform if needed
        
        # Calculate metrics
        metrics = self.calculate_metrics(targets, predictions)
        
        return metrics, predictions, targets


class ForecastVisualizer:
    """Visualize forecasting results."""
    
    def __init__(self, save_dir: str = 'results'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
        
    def plot_predictions(self, y_true: np.ndarray, y_pred: np.ndarray,
                        title: str = "Predictions vs Actual",
                        num_samples: int = 100,
                        save_name: str = "predictions.png"):
        """Plot predictions vs actual values."""
        # Take first num_samples for visualization
        y_true_plot = y_true[:num_samples].flatten()
        y_pred_plot = y_pred[:num_samples].flatten()
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Time series plot
        axes[0].plot(y_true_plot, label='Actual', alpha=0.7)
        axes[0].plot(y_pred_plot, label='Predicted', alpha=0.7)
        axes[0].set_xlabel('Time Steps')
        axes[0].set_ylabel('Occupancy')
        axes[0].set_title(f'{title} - Time Series')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Scatter plot
        axes[1].scatter(y_true_plot, y_pred_plot, alpha=0.5)
        
        # Perfect prediction line
        min_val = min(y_true_plot.min(), y_pred_plot.min())
        max_val = max(y_true_plot.max(), y_pred_plot.max())
        axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        
        axes[1].set_xlabel('Actual Occupancy')
        axes[1].set_ylabel('Predicted Occupancy')
        axes[1].set_title(f'{title} - Scatter Plot')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved plot: {save_name}")
    
    def plot_quantile_predictions(self, y_true: np.ndarray, 
                                 quantile_preds: Dict[float, np.ndarray],
                                 sample_idx: int = 0,
                                 save_name: str = "quantile_predictions.png"):
        """Plot quantile predictions with uncertainty bands."""
        # Extract single sample
        y_true_sample = y_true[sample_idx]
        
        horizon = len(y_true_sample)
        x = np.arange(1, horizon + 1)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot ground truth
        ax.plot(x, y_true_sample, 'ko-', label='Actual', linewidth=2, markersize=6)
        
        # Plot median prediction
        if 0.5 in quantile_preds:
            median = quantile_preds[0.5][sample_idx]
            ax.plot(x, median, 'b-', label='Median Forecast', linewidth=2)
        
        # Plot uncertainty bands
        if 0.1 in quantile_preds and 0.9 in quantile_preds:
            lower = quantile_preds[0.1][sample_idx]
            upper = quantile_preds[0.9][sample_idx]
            ax.fill_between(x, lower, upper, alpha=0.3, label='80% Prediction Interval')
        
        ax.set_xlabel('Forecast Horizon (Days)')
        ax.set_ylabel('Occupancy')
        ax.set_title('Multi-horizon Forecast with Uncertainty')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved plot: {save_name}")
    
    def plot_horizon_performance(self, metrics: Dict[str, float],
                                save_name: str = "horizon_performance.png"):
        """Plot performance metrics across forecast horizons."""
        # Extract per-horizon MAE and RMSE
        horizons = []
        mae_values = []
        rmse_values = []
        
        for key, value in metrics.items():
            if 'MAE_Day' in key:
                day = int(key.split('Day')[1])
                horizons.append(day)
                mae_values.append(value)
            elif 'RMSE_Day' in key:
                day = int(key.split('Day')[1])
                rmse_key = key
                rmse_values.append(value)
        
        if not horizons:
            print("No per-horizon metrics found")
            return
        
        # Sort by horizon
        sorted_data = sorted(zip(horizons, mae_values, rmse_values))
        horizons, mae_values, rmse_values = zip(*sorted_data)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # MAE
        axes[0].plot(horizons, mae_values, 'o-', linewidth=2, markersize=8)
        axes[0].set_xlabel('Forecast Horizon (Days)')
        axes[0].set_ylabel('MAE')
        axes[0].set_title('Mean Absolute Error by Horizon')
        axes[0].grid(True, alpha=0.3)
        
        # RMSE
        axes[1].plot(horizons, rmse_values, 'o-', linewidth=2, color='orange')
        axes[1].set_xlabel('Forecast Horizon (Days)')
        axes[1].set_ylabel('RMSE')
        axes[1].set_title('Root Mean Squared Error by Horizon')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved plot: {save_name}")
    
    def plot_attention_weights(self, attention_weights: torch.Tensor,
                              save_name: str = "attention_weights.png"):
        """Visualize attention weights."""
        # Take first sample, average across heads
        if len(attention_weights.shape) == 4:
            # [batch, heads, time, time]
            attention = attention_weights[0].mean(dim=0).cpu().numpy()
        else:
            attention = attention_weights[0].cpu().numpy()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(attention, cmap='viridis', ax=ax, cbar_kws={'label': 'Attention Weight'})
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
        ax.set_title('Self-Attention Weights')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved plot: {save_name}")
    
    def plot_feature_importance(self, feature_weights: torch.Tensor,
                               feature_names: List[str],
                               save_name: str = "feature_importance.png"):
        """Plot feature importance from variable selection."""
        # Average over batch and time
        importance = feature_weights.mean(dim=(0, 1)).cpu().numpy()
        
        # Sort
        sorted_idx = np.argsort(importance)[::-1]
        sorted_importance = importance[sorted_idx]
        sorted_names = [feature_names[i] for i in sorted_idx]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.barh(range(len(sorted_names)), sorted_importance)
        ax.set_yticks(range(len(sorted_names)))
        ax.set_yticklabels(sorted_names)
        ax.set_xlabel('Importance')
        ax.set_title('Feature Importance (Variable Selection Network)')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved plot: {save_name}")
    
    def create_evaluation_report(self, metrics: Dict[str, float],
                                save_name: str = "evaluation_report.txt"):
        """Create a text report of evaluation metrics."""
        report_path = self.save_dir / save_name
        
        with open(report_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("TFT Evaluation Report\n")
            f.write("=" * 60 + "\n\n")
            
            # Overall metrics
            f.write("Overall Metrics:\n")
            f.write("-" * 40 + "\n")
            for key in ['MAE', 'RMSE', 'MAPE', 'R2']:
                if key in metrics:
                    f.write(f"{key:20s}: {metrics[key]:10.4f}\n")
            
            f.write("\n")
            
            # Quantile metrics
            f.write("Quantile Metrics:\n")
            f.write("-" * 40 + "\n")
            for key, value in metrics.items():
                if 'Pinball' in key or 'Coverage' in key:
                    f.write(f"{key:20s}: {value:10.4f}\n")
            
            f.write("\n")
            
            # Per-horizon metrics
            f.write("Per-Horizon MAE:\n")
            f.write("-" * 40 + "\n")
            for key, value in sorted(metrics.items()):
                if 'MAE_Day' in key:
                    day = key.split('Day')[1]
                    f.write(f"Day {day:2s}: {value:10.4f}\n")
        
        print(f"✓ Saved report: {save_name}")


def main():
    """Example usage of evaluation tools."""
    # Generate dummy predictions for demonstration
    num_samples = 100
    horizon = 15
    
    y_true = np.random.randn(num_samples, horizon) * 10 + 50
    y_pred = y_true + np.random.randn(num_samples, horizon) * 2
    
    # Evaluate
    evaluator = ForecastEvaluator()
    metrics = evaluator.calculate_metrics(y_true, y_pred)
    
    print("\nMetrics:")
    for key, value in metrics.items():
        if not 'Day' in key:  # Print only overall metrics
            print(f"  {key}: {value:.4f}")
    
    # Visualize
    visualizer = ForecastVisualizer(save_dir='results')
    visualizer.plot_predictions(y_true, y_pred)
    visualizer.plot_horizon_performance(metrics)
    visualizer.create_evaluation_report(metrics)
    
    print("\n✓ Evaluation complete!")


if __name__ == "__main__":
    main()
