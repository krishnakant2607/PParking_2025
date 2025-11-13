"""
Temporal Fusion Transformer (TFT) implementation for parking occupancy forecasting.
Based on: "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional


class GatedResidualNetwork(nn.Module):
    """Gated Residual Network (GRN) for feature processing."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 dropout: float = 0.1, context_dim: Optional[int] = None):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        
        # Primary layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Context processing
        if context_dim is not None:
            self.context_fc = nn.Linear(context_dim, hidden_dim, bias=False)
        
        # Gate for controlling feature contribution
        self.gate_fc = nn.Linear(hidden_dim, output_dim)
        
        # Output projection
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        
        # Skip connection
        if input_dim != output_dim:
            self.skip_fc = nn.Linear(input_dim, output_dim)
        else:
            self.skip_fc = None
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)
        
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Primary path
        a = self.fc1(x)
        
        # Add context if provided
        if context is not None and self.context_dim is not None:
            a = a + self.context_fc(context)
        
        a = F.elu(a)
        a = self.fc2(a)
        a = self.dropout(a)
        
        # Gating mechanism
        g = torch.sigmoid(self.gate_fc(a))
        
        # Output
        out = self.fc_out(a)
        out = g * out
        
        # Skip connection
        if self.skip_fc is not None:
            skip = self.skip_fc(x)
        else:
            skip = x
        
        # Residual connection and normalization
        out = self.layer_norm(out + skip)
        
        return out


class VariableSelectionNetwork(nn.Module):
    """Variable Selection Network for feature importance."""
    
    def __init__(self, input_dim: int, num_features: int, hidden_dim: int, 
                 dropout: float = 0.1, context_dim: Optional[int] = None):
        super().__init__()
        
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        
        # Feature-specific GRNs
        self.feature_grns = nn.ModuleList([
            GatedResidualNetwork(input_dim, hidden_dim, hidden_dim, dropout)
            for _ in range(num_features)
        ])
        
        # Concatenated features processing
        self.concat_grn = GatedResidualNetwork(
            num_features * hidden_dim, 
            hidden_dim, 
            num_features,
            dropout,
            context_dim
        )
        
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, time, num_features, input_dim]
            context: Optional context vector
            
        Returns:
            selected_features: [batch, time, hidden_dim]
            weights: [batch, time, num_features]
        """
        batch_size, time_steps, num_features, _ = x.shape
        
        # Process each feature
        feature_outputs = []
        for i in range(num_features):
            feature_i = x[:, :, i, :]  # [batch, time, input_dim]
            processed = self.feature_grns[i](feature_i)  # [batch, time, hidden_dim]
            feature_outputs.append(processed)
        
        # Stack features
        stacked = torch.stack(feature_outputs, dim=2)  # [batch, time, num_features, hidden_dim]
        
        # Flatten for concatenation
        flattened = stacked.reshape(batch_size, time_steps, -1)  # [batch, time, num_features * hidden_dim]
        
        # Compute feature weights
        weights = self.concat_grn(flattened, context)  # [batch, time, num_features]
        weights = self.softmax(weights)
        
        # Apply weights
        weights_expanded = weights.unsqueeze(-1)  # [batch, time, num_features, 1]
        selected = (stacked * weights_expanded).sum(dim=2)  # [batch, time, hidden_dim]
        
        return selected, weights


class InterpretableMultiHeadAttention(nn.Module):
    """Multi-head attention with interpretability."""
    
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.q_linear = nn.Linear(hidden_dim, hidden_dim)
        self.k_linear = nn.Linear(hidden_dim, hidden_dim)
        self.v_linear = nn.Linear(hidden_dim, hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query, key, value: [batch, time, hidden_dim]
            mask: Optional attention mask
            
        Returns:
            output: [batch, time, hidden_dim]
            attention: [batch, num_heads, time, time]
        """
        batch_size = query.shape[0]
        
        # Linear projections
        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scale = self.scale.to(query.device)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Attention weights
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # Apply attention to values
        context = torch.matmul(attention, V)
        
        # Reshape and project
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)
        output = self.out_linear(context)
        
        return output, attention


class TemporalFusionTransformer(nn.Module):
    """
    Temporal Fusion Transformer for multi-horizon forecasting.
    
    Architecture:
    1. Variable Selection Networks (static, historical, future)
    2. LSTM Encoder-Decoder
    3. Multi-head Self-Attention
    4. Position-wise Feed-Forward
    5. Quantile Output Layer
    """
    
    def __init__(self, 
                 static_features: int = 0,
                 historical_features: int = 10,
                 future_features: int = 5,
                 hidden_dim: int = 160,
                 num_heads: int = 4,
                 num_encoder_steps: int = 30,
                 num_decoder_steps: int = 15,
                 dropout: float = 0.1,
                 num_quantiles: int = 3):
        super().__init__()
        
        self.static_features = static_features
        self.historical_features = historical_features
        self.future_features = future_features
        self.hidden_dim = hidden_dim
        self.num_encoder_steps = num_encoder_steps
        self.num_decoder_steps = num_decoder_steps
        self.num_quantiles = num_quantiles
        
        # Variable Selection Networks
        if static_features > 0:
            self.static_vsn = VariableSelectionNetwork(
                1, static_features, hidden_dim, dropout
            )
        
        self.historical_vsn = VariableSelectionNetwork(
            1, historical_features, hidden_dim, dropout
        )
        
        if future_features > 0:
            self.future_vsn = VariableSelectionNetwork(
                1, future_features, hidden_dim, dropout
            )
        
        # LSTM Encoder-Decoder
        self.encoder_lstm = nn.LSTM(
            hidden_dim, hidden_dim, num_layers=1, batch_first=True, dropout=dropout
        )
        
        self.decoder_lstm = nn.LSTM(
            hidden_dim, hidden_dim, num_layers=1, batch_first=True, dropout=dropout
        )
        
        # Static enrichment (if static features exist)
        if static_features > 0:
            self.static_context_grn = GatedResidualNetwork(
                hidden_dim, hidden_dim, hidden_dim, dropout, context_dim=hidden_dim
            )
        
        # Temporal Self-Attention
        self.attention = InterpretableMultiHeadAttention(hidden_dim, num_heads, dropout)
        self.attention_grn = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)
        
        # Position-wise Feed-Forward
        self.ff_grn = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)
        
        # Output layer (quantile forecasting)
        self.output_layer = nn.Linear(hidden_dim, num_quantiles)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, 
                historical_inputs: torch.Tensor,
                future_inputs: Optional[torch.Tensor] = None,
                static_inputs: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            historical_inputs: [batch, encoder_steps, historical_features]
            future_inputs: [batch, decoder_steps, future_features]
            static_inputs: [batch, static_features]
            
        Returns:
            predictions: [batch, decoder_steps, num_quantiles]
            interpretability_outputs: Dictionary with attention weights and feature importances
        """
        batch_size = historical_inputs.shape[0]
        device = historical_inputs.device
        
        interpretability = {}
        
        # Reshape inputs for VSN: [batch, time, num_features, 1]
        historical_reshaped = historical_inputs.unsqueeze(-1)
        
        # Historical Variable Selection
        historical_selected, historical_weights = self.historical_vsn(historical_reshaped)
        interpretability['historical_weights'] = historical_weights
        
        # Encode historical context
        encoder_output, (h_n, c_n) = self.encoder_lstm(historical_selected)
        
        # Static context enrichment (if applicable)
        if static_inputs is not None and self.static_features > 0:
            static_reshaped = static_inputs.unsqueeze(1).unsqueeze(-1)  # [batch, 1, features, 1]
            static_selected, static_weights = self.static_vsn(static_reshaped)
            interpretability['static_weights'] = static_weights
            
            static_context = static_selected.squeeze(1)  # [batch, hidden_dim]
            
            # Enrich encoder output with static context
            encoder_output = self.static_context_grn(encoder_output, static_context.unsqueeze(1))
        
        # Future inputs (known future features like day of week, etc.)
        if future_inputs is not None and self.future_features > 0:
            future_reshaped = future_inputs.unsqueeze(-1)
            future_selected, future_weights = self.future_vsn(future_reshaped)
            interpretability['future_weights'] = future_weights
        else:
            # Create dummy future inputs if not provided
            future_selected = torch.zeros(
                batch_size, self.num_decoder_steps, self.hidden_dim
            ).to(device)
        
        # Decode with LSTM
        decoder_output, _ = self.decoder_lstm(future_selected, (h_n, c_n))
        
        # Temporal Self-Attention
        # Combine encoder and decoder outputs
        combined = torch.cat([encoder_output, decoder_output], dim=1)
        
        # Self-attention on combined sequence
        attended, attention_weights = self.attention(combined, combined, combined)
        interpretability['attention_weights'] = attention_weights
        
        # Take only decoder part
        attended_decoder = attended[:, self.num_encoder_steps:, :]
        
        # Apply GRN after attention
        attended_decoder = self.attention_grn(attended_decoder)
        
        # Position-wise Feed-Forward
        output = self.ff_grn(attended_decoder)
        
        # Quantile predictions
        predictions = self.output_layer(output)  # [batch, decoder_steps, num_quantiles]
        
        return predictions, interpretability


class QuantileLoss(nn.Module):
    """Quantile loss for probabilistic forecasting."""
    
    def __init__(self, quantiles: list = [0.1, 0.5, 0.9]):
        super().__init__()
        self.quantiles = quantiles
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: [batch, time, num_quantiles]
            targets: [batch, time]
            
        Returns:
            loss: Scalar quantile loss
        """
        losses = []
        
        for i, q in enumerate(self.quantiles):
            errors = targets - predictions[:, :, i]
            loss = torch.max((q - 1) * errors, q * errors)
            losses.append(loss)
        
        return torch.mean(torch.stack(losses))


def main():
    """Test the TFT model."""
    # Model parameters
    batch_size = 32
    encoder_steps = 30
    decoder_steps = 15
    historical_features = 10
    future_features = 5
    
    # Create dummy data
    historical_inputs = torch.randn(batch_size, encoder_steps, historical_features)
    future_inputs = torch.randn(batch_size, decoder_steps, future_features)
    targets = torch.randn(batch_size, decoder_steps)
    
    # Initialize model
    model = TemporalFusionTransformer(
        static_features=0,
        historical_features=historical_features,
        future_features=future_features,
        hidden_dim=160,
        num_heads=4,
        num_encoder_steps=encoder_steps,
        num_decoder_steps=decoder_steps,
        dropout=0.1,
        num_quantiles=3
    )
    
    # Forward pass
    predictions, interpretability = model(historical_inputs, future_inputs)
    
    print(f"Input shape: {historical_inputs.shape}")
    print(f"Future input shape: {future_inputs.shape}")
    print(f"Output shape: {predictions.shape}")
    print(f"\nInterpretability outputs:")
    for key, value in interpretability.items():
        print(f"  {key}: {value.shape}")
    
    # Test loss
    criterion = QuantileLoss(quantiles=[0.1, 0.5, 0.9])
    loss = criterion(predictions, targets)
    print(f"\nQuantile loss: {loss.item():.4f}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")


if __name__ == "__main__":
    main()
