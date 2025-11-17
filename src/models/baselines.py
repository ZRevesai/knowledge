"""
Baseline models for comparison with KGNN.

Implements:
1. Standard Feed-Forward Neural Network (FFNN)
2. XGBoost model
3. Transformer-based model
4. FFNN with SHAP explanations
5. LIME-based model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings

# Try to import optional dependencies
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not installed. XGBoostModel will not be available.")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not installed. FFNNWithSHAP will have limited functionality.")

try:
    from lime import lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    warnings.warn("LIME not installed. LIMEModel will have limited functionality.")


class StandardFFNN(nn.Module):
    """
    Standard Feed-Forward Neural Network without knowledge integration.

    Baseline model for comparison with KGNN.
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int] = [256, 128, 64],
                 num_outputs: int = 12,
                 dropout_rate: float = 0.3,
                 activation: str = 'relu',
                 use_batch_norm: bool = True):
        """
        Initialize standard FFNN.

        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
            num_outputs: Number of output neurons
            dropout_rate: Dropout probability
            activation: Activation function
            use_batch_norm: Whether to use batch normalization
        """
        super(StandardFFNN, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_outputs = num_outputs

        # Build layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))

            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'elu':
                layers.append(nn.ELU())

            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, num_outputs))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Predictions of shape (batch_size, num_outputs)
        """
        return self.network(x)


class XGBoostModel:
    """
    XGBoost model for micronutrient deficiency detection.

    Uses gradient boosting with feature importance capabilities.
    """

    def __init__(self,
                 num_outputs: int = 12,
                 max_depth: int = 6,
                 learning_rate: float = 0.1,
                 n_estimators: int = 100,
                 random_state: int = 42):
        """
        Initialize XGBoost model.

        Args:
            num_outputs: Number of outputs (micronutrients)
            max_depth: Maximum tree depth
            learning_rate: Learning rate
            n_estimators: Number of boosting rounds
            random_state: Random seed
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not installed. Install with: pip install xgboost")

        self.num_outputs = num_outputs
        self.models = []

        # Create separate model for each output
        for _ in range(num_outputs):
            model = xgb.XGBClassifier(
                max_depth=max_depth,
                learning_rate=learning_rate,
                n_estimators=n_estimators,
                random_state=random_state,
                objective='binary:logistic',
                use_label_encoder=False,
                eval_metric='logloss'
            )
            self.models.append(model)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train the model.

        Args:
            X: Training features of shape (n_samples, n_features)
            y: Training labels of shape (n_samples, num_outputs)
        """
        for i, model in enumerate(self.models):
            model.fit(X, y[:, i])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Input features of shape (n_samples, n_features)

        Returns:
            Predictions of shape (n_samples, num_outputs)
        """
        predictions = []
        for model in self.models:
            pred = model.predict_proba(X)[:, 1]
            predictions.append(pred)

        return np.column_stack(predictions)

    def get_feature_importance(self) -> Dict[int, np.ndarray]:
        """
        Get feature importance for each output.

        Returns:
            Dictionary mapping output index to feature importance array
        """
        importance_dict = {}
        for i, model in enumerate(self.models):
            importance_dict[i] = model.feature_importances_

        return importance_dict


class TransformerModel(nn.Module):
    """
    Transformer-based model for nutritional assessment.

    Uses multi-head self-attention mechanisms adapted for tabular data.
    """

    def __init__(self,
                 input_dim: int,
                 d_model: int = 128,
                 nhead: int = 4,
                 num_layers: int = 2,
                 dim_feedforward: int = 256,
                 num_outputs: int = 12,
                 dropout: float = 0.3):
        """
        Initialize Transformer model.

        Args:
            input_dim: Number of input features
            d_model: Dimension of transformer model
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Dimension of feedforward network
            num_outputs: Number of outputs
            dropout: Dropout probability
        """
        super(TransformerModel, self).__init__()

        self.input_dim = input_dim
        self.d_model = d_model

        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)

        # Positional encoding (for feature order)
        self.positional_encoding = nn.Parameter(torch.randn(1, 1, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, num_outputs),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Predictions of shape (batch_size, num_outputs)
        """
        # Project input to d_model dimension
        x = self.input_projection(x)

        # Add positional encoding
        x = x.unsqueeze(1)  # (batch_size, 1, d_model)
        x = x + self.positional_encoding

        # Apply transformer
        transformed = self.transformer(x)

        # Pool and predict
        pooled = transformed.squeeze(1)  # (batch_size, d_model)
        output = self.output_layer(pooled)

        return output


class FFNNWithSHAP:
    """
    Standard FFNN with SHAP explanations.

    Combines black-box FFNN with post-hoc SHAP interpretability.
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int] = [256, 128, 64],
                 num_outputs: int = 12,
                 dropout_rate: float = 0.3):
        """
        Initialize FFNN with SHAP.

        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
            num_outputs: Number of outputs
            dropout_rate: Dropout probability
        """
        self.model = StandardFFNN(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            num_outputs=num_outputs,
            dropout_rate=dropout_rate
        )
        self.explainer = None
        self.background_data = None

    def fit(self,
            X_train: torch.Tensor,
            y_train: torch.Tensor,
            num_epochs: int = 100,
            batch_size: int = 32,
            learning_rate: float = 0.001):
        """
        Train the model.

        Args:
            X_train: Training features
            y_train: Training labels
            num_epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
        """
        # Train FFNN
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.BCELoss()

        self.model.train()
        for epoch in range(num_epochs):
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        # Initialize SHAP explainer
        if SHAP_AVAILABLE:
            # Use subset of training data as background
            indices = np.random.choice(len(X_train), min(100, len(X_train)), replace=False)
            self.background_data = X_train[indices]

            # Create explainer
            self.explainer = shap.DeepExplainer(self.model, self.background_data)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Make predictions.

        Args:
            X: Input features

        Returns:
            Predictions
        """
        self.model.eval()
        with torch.no_grad():
            return self.model(X)

    def explain(self, X: torch.Tensor) -> Optional[np.ndarray]:
        """
        Get SHAP explanations for predictions.

        Args:
            X: Input features

        Returns:
            SHAP values array or None if SHAP not available
        """
        if not SHAP_AVAILABLE or self.explainer is None:
            warnings.warn("SHAP explainer not available")
            return None

        shap_values = self.explainer.shap_values(X)
        return shap_values


class LIMEModel:
    """
    Model with LIME (Local Interpretable Model-agnostic Explanations).

    Wraps any model with LIME for local interpretability.
    """

    def __init__(self,
                 base_model,
                 feature_names: Optional[List[str]] = None,
                 class_names: Optional[List[str]] = None):
        """
        Initialize LIME model.

        Args:
            base_model: Base model to wrap
            feature_names: Names of input features
            class_names: Names of output classes
        """
        self.base_model = base_model
        self.feature_names = feature_names
        self.class_names = class_names
        self.explainer = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Train the base model and initialize LIME explainer.

        Args:
            X_train: Training features
            y_train: Training labels
        """
        # Train base model
        if hasattr(self.base_model, 'fit'):
            self.base_model.fit(X_train, y_train)

        # Initialize LIME explainer
        if LIME_AVAILABLE:
            self.explainer = lime_tabular.LimeTabularExplainer(
                training_data=X_train,
                feature_names=self.feature_names,
                class_names=self.class_names,
                mode='classification'
            )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Input features

        Returns:
            Predictions
        """
        if isinstance(self.base_model, nn.Module):
            self.base_model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X)
                return self.base_model(X_tensor).numpy()
        else:
            return self.base_model.predict(X)

    def explain_instance(self,
                        instance: np.ndarray,
                        num_features: int = 10) -> Optional[Dict]:
        """
        Explain a single prediction using LIME.

        Args:
            instance: Single input instance
            num_features: Number of features to include in explanation

        Returns:
            Explanation dictionary or None if LIME not available
        """
        if not LIME_AVAILABLE or self.explainer is None:
            warnings.warn("LIME explainer not available")
            return None

        # Define prediction function for LIME
        def predict_fn(X):
            return self.predict(X)

        # Get explanation
        explanation = self.explainer.explain_instance(
            data_row=instance,
            predict_fn=predict_fn,
            num_features=num_features
        )

        return {
            'explanation': explanation,
            'feature_importance': dict(explanation.as_list())
        }


if __name__ == "__main__":
    print("Testing Baseline Models...")

    input_dim = 105
    num_outputs = 12
    batch_size = 32

    # Test Standard FFNN
    print("\n1. Testing Standard FFNN...")
    ffnn = StandardFFNN(input_dim=input_dim, num_outputs=num_outputs)
    x = torch.randn(batch_size, input_dim)
    output = ffnn(x)
    print(f"   Input: {x.shape}, Output: {output.shape}")

    # Test XGBoost
    if XGBOOST_AVAILABLE:
        print("\n2. Testing XGBoost Model...")
        xgb_model = XGBoostModel(num_outputs=num_outputs, n_estimators=10)
        X_train = np.random.randn(100, input_dim)
        y_train = np.random.randint(0, 2, (100, num_outputs))
        xgb_model.fit(X_train, y_train)
        predictions = xgb_model.predict(X_train[:10])
        print(f"   Training: {X_train.shape}, Predictions: {predictions.shape}")

    # Test Transformer
    print("\n3. Testing Transformer Model...")
    transformer = TransformerModel(input_dim=input_dim, num_outputs=num_outputs)
    output = transformer(x)
    print(f"   Input: {x.shape}, Output: {output.shape}")

    # Test FFNN with SHAP
    print("\n4. Testing FFNN with SHAP...")
    ffnn_shap = FFNNWithSHAP(input_dim=input_dim, num_outputs=num_outputs)
    print(f"   Model created with SHAP support: {SHAP_AVAILABLE}")

    # Test LIME Model
    print("\n5. Testing LIME Model...")
    base_model = StandardFFNN(input_dim=input_dim, num_outputs=num_outputs)
    lime_model = LIMEModel(base_model)
    print(f"   Model created with LIME support: {LIME_AVAILABLE}")

    print("\nBaseline models test completed!")
