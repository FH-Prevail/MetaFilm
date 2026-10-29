"""MetaFiLM: metadata-conditioned FiLM modulation for cold-start demand forecasting.

MetaFiLM forecasts demand for item-location series that have little or no
sales history by conditioning a compact sequence model on item and location
metadata. Metadata enters through two complementary pathways:

* a **level head**, a small MLP that predicts a base demand level and adds it
  to every forecast step, and
* a bounded **FiLM** (Feature-wise Linear Modulation) block that scales and
  shifts the decoder's hidden states feature by feature.

A two-layer LSTM encodes whatever history is available (it is skipped for a
series with no history), and a one-layer LSTM decoder reads the future
calendar. The encoded history is added to the decoder outputs with weight 0.3
before the FiLM block.

The main entry points are :class:`MetaFiLMConfig` and
:class:`MetaFiLMForecaster`; :class:`MetaFiLMModel` is the underlying
``torch.nn.Module``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

__all__ = [
    "MetaFiLMForecaster",
    "MetaFiLMConfig",
    "MetaFiLMModel",
    "StabilizedFiLMLayer",
    "TimeSeriesDataset",
    "ValidationError",
    "ModelNotFittedError"
]


class ValidationError(Exception):
    """Raised for an invalid configuration or input data frame."""


class ModelNotFittedError(Exception):
    """Raised when predicting with, or saving, a forecaster that is not fitted."""


@dataclass
class MetaFiLMConfig:
    """Configuration for :class:`MetaFiLMForecaster`.

    Data
        n_steps_lag: length of the history window fed to the encoder.
        forecast_horizon: number of steps forecast at once.
        context_features: metadata columns (categorical or numeric), constant
            per series. Categorical values unseen at fit time map to a reserved
            "unknown" code, so new items and locations are handled.
        temporal_features: optional extra per-step columns fed to the encoder
            next to the target.
        item_id_col, target_col, date_col: column names of the long-format
            input frame (one row per series and period).
        freq: pandas offset alias of the series, e.g. ``"W-SUN"`` (weekly) or
            ``"MS"`` (monthly); used to build the future calendar.

    Model
        hidden_dim, num_lstm_layers, dropout: network size.
        use_residual: add ``0.5 * features`` after the FiLM modulation.
        film_mode: ``"bounded"`` (default, gamma = 1 + 0.1 tanh, beta = 0.1 tanh),
            ``"unbounded"`` (classic FiLM), ``"concat"`` (parameter-matched
            concatenation control) or ``"off"`` (no FiLM).
        film_init: ``"default"`` or ``"identity"`` (FiLM generator starts at
            gamma = 1, beta = 0).
        use_level, use_history: switch the level head or the history encoder
            off (used for ablations).

    Training
        loss: ``"mse"`` (default), ``"l1"`` (mean absolute error) or ``"huber"``.
            When forecasts are evaluated by MAE, ``"l1"`` or ``"huber"``
            usually lowers the error; ``"huber"`` is the more balanced choice
            for intermittent demand.
        batch_size, learning_rate, num_epochs, patience, weight_decay: optimiser
            and early-stopping settings. Weight decay applies to the metadata
            fusion block only.
        restore_best_weights: after early stopping, restore the weights of the
            epoch with the lowest validation loss (default). ``False`` keeps
            the final-epoch weights, which is how the CENTERIS 2026 paper's
            experiments were run.
        random_seed, device: reproducibility and placement.
    """
    # Data configuration
    n_steps_lag: int = 12
    forecast_horizon: int = 12
    context_features: List[str] = None
    temporal_features: List[str] = None
    item_id_col: str = "compoundkey"
    target_col: str = "quantity"
    date_col: str = "orderweek_dt"
    freq: str = "W-SUN"  # pandas offset alias; e.g. "W-SUN" weekly, "MS" monthly

    # Model architecture
    hidden_dim: int = 128
    num_lstm_layers: int = 2
    dropout: float = 0.3
    use_residual: bool = True
    # FiLM ablation knobs:
    #   film_mode="bounded"   -> gamma=1+0.1*tanh(.), beta=0.1*tanh(.)  (default/paper)
    #   film_mode="unbounded" -> gamma=1+gamma_raw,    beta=beta_raw     (classic FiLM)
    #   film_mode="concat"    -> replace FiLM with parameter-matched concat + MLP fusion
    #   film_mode="off"       -> identity passthrough (FiLM removed entirely)
    film_mode: str = "bounded"
    film_init: str = "default"   # "default" | "identity" (zero-init FiLM generator)
    # Component ablation switches (default True = full model):
    use_level: bool = True       # metadata level head adds a base demand offset
    use_history: bool = True     # LSTM history encoder feeds the decoder
    # Training objective: "mse" (paper) | "l1" (MAE-aligned) | "huber"
    loss: str = "mse"
    # Restore the best-validation weights after early stopping. The paper's
    # experiments kept the final-epoch weights (experiments/common.py passes
    # False to reproduce them exactly).
    restore_best_weights: bool = True

    # Training parameters
    batch_size: int = 32
    learning_rate: float = 0.001
    num_epochs: int = 100
    patience: int = 10
    weight_decay: float = 0.01
    random_seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        if self.context_features is None:
            self.context_features = []
        if self.temporal_features is None:
            self.temporal_features = []
        if self.n_steps_lag <= 0:
            raise ValidationError("n_steps_lag must be positive")
        if self.forecast_horizon <= 0:
            raise ValidationError("forecast_horizon must be positive")
        if self.film_mode not in ("bounded", "unbounded", "concat", "off"):
            raise ValidationError(
                "film_mode must be 'bounded', 'unbounded', 'concat', or 'off'"
            )
        if self.film_init not in ("default", "identity"):
            raise ValidationError("film_init must be 'default' or 'identity'")
        if self.loss not in ("mse", "l1", "huber"):
            raise ValidationError("loss must be 'mse', 'l1', or 'huber'")


class StabilizedFiLMLayer(nn.Module):
    """Feature-wise linear modulation of decoder states by metadata.

    In the default ``"bounded"`` mode the generator output is squashed so that
    gamma = 1 + 0.1 tanh(.) and beta = 0.1 tanh(.): conditioning starts close
    to the identity and only grows if training supports it. See
    :class:`MetaFiLMConfig` for the other modes.
    """

    def __init__(self, feature_dim: int, context_dim: int, use_residual: bool = True,
                 film_mode: str = "bounded", film_init: str = "default"):
        super().__init__()
        self.use_residual = use_residual
        self.film_mode = film_mode

        if film_mode == "concat":
            # Reviewer-requested late-fusion control.  Its parameter count is
            # approximately matched to the FiLM generator: concatenate the
            # same context to every decoder state, then map back to feature_dim.
            self.concat_fusion = nn.Sequential(
                nn.Linear(feature_dim + context_dim, feature_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(feature_dim, feature_dim),
            )
            self.film_generator = None
        else:
            self.concat_fusion = None
            self.film_generator = nn.Sequential(
                nn.Linear(context_dim, feature_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(feature_dim, 2 * feature_dim)
            )

        # "identity" init: zero the final layer so gamma_raw=beta_raw=0 at start,
        # i.e. gamma=1, beta=0 regardless of film_mode. This lets an *unbounded*
        # FiLM start exactly at identity (matching bounded's starting point), so
        # an unbounded-vs-bounded comparison isn't confounded by initial scale.
        if film_init == "identity" and self.film_generator is not None:
            last_linear = self.film_generator[-1]
            with torch.no_grad():
                last_linear.weight.zero_()
                last_linear.bias.zero_()

    def forward(self, features: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Apply (optionally ablated) FiLM modulation.

        Args:
            features: (batch, feature_dim) or (batch, seq, feature_dim)
            context: (batch, context_dim)
        """
        # "off": FiLM removed entirely -> identity passthrough (no modulation,
        # no residual). This is the clean "no-FiLM" ablation arm.
        if self.film_mode == "off":
            return features

        if self.film_mode == "concat":
            if features.dim() == 3:
                expanded_context = context.unsqueeze(1).expand(-1, features.shape[1], -1)
            else:
                expanded_context = context
            return self.concat_fusion(torch.cat([features, expanded_context], dim=-1))

        film_params = self.film_generator(context)
        gamma_raw, beta_raw = film_params.chunk(2, dim=-1)

        if self.film_mode == "unbounded":
            # Classic FiLM: unconstrained affine, identity-init friendly.
            gamma = 1.0 + gamma_raw
            beta = beta_raw
        else:  # "bounded" (default / paper)
            # Stabilize: anchor near identity
            gamma = 1.0 + 0.1 * torch.tanh(gamma_raw)
            beta = 0.1 * torch.tanh(beta_raw)

        # Expand for sequence if needed
        if features.dim() == 3:
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)

        # Apply FiLM
        modulated = gamma * features + beta

        # Weighted residual
        if self.use_residual:
            modulated = modulated + 0.5 * features

        return modulated


class MetaFiLMModel(nn.Module):
    """The MetaFiLM network.

    1. Level head: base demand level from metadata, added to every step.
    2. History encoder: two-layer LSTM over the available past (skipped when a
       series has no history).
    3. Future decoder: one-layer LSTM over the future calendar, zero initial
       state; the encoded history is added to its outputs with weight 0.3.
    4. FiLM block: metadata scales and shifts the fused decoder states.
    """

    def __init__(
        self,
        past_feature_dim: int,
        future_feature_dim: int,
        context_dim: int,
        hidden_dim: int = 128,
        num_lstm_layers: int = 2,
        forecast_horizon: int = 12,
        dropout: float = 0.3,
        use_residual: bool = True,
        film_mode: str = "bounded",
        film_init: str = "default",
        use_level: bool = True,
        use_history: bool = True
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.forecast_horizon = forecast_horizon
        self.use_level = use_level
        self.use_history = use_history

        # Level prediction from metadata (zero-shot base)
        self.level_head = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

        # History encoder (optional)
        self.history_encoder = nn.LSTM(
            input_size=past_feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0
        )

        # Future decoder
        self.future_decoder = nn.LSTM(
            input_size=future_feature_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

        # FiLM for future patterns
        self.film_layer = StabilizedFiLMLayer(
            feature_dim=hidden_dim,
            context_dim=context_dim,
            use_residual=use_residual,
            film_mode=film_mode,
            film_init=film_init
        )

        # Vectorized output head
        self.output_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        context: torch.Tensor,
        future_features: torch.Tensor,
        past_sequence: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with vectorized output.

        Args:
            context: (batch, context_dim)
            future_features: (batch, horizon, future_feature_dim)
            past_sequence: (batch, seq_len, past_feature_dim) or None

        Returns:
            predictions: (batch, horizon)
        """
        batch_size = context.shape[0]

        # 1. Base level from metadata
        if self.use_level:
            base_level = self.level_head(context)  # (batch, 1)
        else:
            base_level = torch.zeros(batch_size, 1, device=context.device)

        # 2. Encode history if available (and enabled)
        if self.use_history and past_sequence is not None:
            _, (h_past, _) = self.history_encoder(past_sequence)
            history_context = h_past[-1]  # (batch, hidden_dim)
        else:
            history_context = torch.zeros(batch_size, self.hidden_dim, device=context.device)

        # 3. Decode future
        future_encoded, _ = self.future_decoder(future_features)  # (batch, horizon, hidden_dim)

        # 4. Combine with history context
        combined = future_encoded + 0.3 * history_context.unsqueeze(1)

        # 5. Apply FiLM modulation (vectorized)
        modulated = self.film_layer(combined, context)  # (batch, horizon, hidden_dim)

        # 6. Vectorized output
        predictions = self.output_head(modulated).squeeze(-1)  # (batch, horizon)

        # 7. Add base level
        predictions = predictions + base_level

        return predictions


class TimeSeriesDataset(Dataset):
    """Torch dataset of (context, future calendar, past window, target) samples.

    The past window may be ``None`` for series without history.
    """

    def __init__(
        self,
        context: np.ndarray,
        future_features: np.ndarray,
        past_sequences: Optional[List[Optional[np.ndarray]]] = None,
        targets: Optional[np.ndarray] = None
    ):
        self.context = torch.FloatTensor(context)
        self.future_features = torch.FloatTensor(future_features)
        self.past_sequences = past_sequences  # Keep as list
        self.targets = torch.FloatTensor(targets) if targets is not None else None

    def __len__(self):
        return len(self.context)

    def __getitem__(self, idx):
        items = [self.context[idx], self.future_features[idx]]

        if self.past_sequences is not None:
            past = self.past_sequences[idx]
            if past is not None:
                items.append(torch.FloatTensor(past))
            else:
                items.append(None)

        if self.targets is not None:
            items.append(self.targets[idx])

        return tuple(items)


def _collate_train(batch):
    """Custom collate that asserts past sequences exist for training/validation."""
    ctx, fut, pst, tgt = zip(*batch)
    assert all(p is not None for p in pst), "Training past sequences must be present"
    return (
        torch.stack(ctx, dim=0),
        torch.stack(fut, dim=0),
        torch.stack(pst, dim=0),
        torch.stack(tgt, dim=0),
    )


class MetaFiLMForecaster(BaseEstimator):
    """Fit and apply a MetaFiLM model on a long-format demand panel.

    The input frame has one row per series and period, with the columns named
    in the config: an item id, a date, the target and the metadata columns
    (``context_features``, constant within a series). Training uses every
    window of ``n_steps_lag + forecast_horizon`` periods; series shorter than
    that are skipped during training but can still be forecast.

    Example::

        from metafilm import MetaFiLMConfig, MetaFiLMForecaster

        config = MetaFiLMConfig(
            n_steps_lag=8, forecast_horizon=4,
            context_features=["category", "store"],
            item_id_col="series_id", date_col="week", target_col="sales",
            freq="W-SUN",
        )
        model = MetaFiLMForecaster(config).fit(train_df)
        forecasts = model.predict(new_items_df)   # {series_id: array(horizon)}
    """

    def __init__(self, config: MetaFiLMConfig):
        self.config = config

        # Set seeds for reproducibility
        torch.manual_seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)

        # Reproducibility settings
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.is_fitted_ = False
        self.model_: Optional[MetaFiLMModel] = None
        self.context_scaler_: Optional[StandardScaler] = None
        self.target_scaler_: Optional[StandardScaler] = None
        self.past_scaler_: Optional[StandardScaler] = None
        self.future_scaler_: Optional[StandardScaler] = None

        self.train_losses_: List[float] = []
        self.val_losses_: List[float] = []

        # Persistent categorical categories learned at fit-time
        self._context_categories: Dict[str, List[str]] = {}

        logger.info(f"Initialized MetaFiLM on device: {self.config.device}")

    def _create_future_features(
        self,
        start_date: pd.Timestamp,
        horizon: int
    ) -> np.ndarray:
        """Calendar features (sin/cos of month and ISO week) for the horizon."""
        future_dates = pd.date_range(start=start_date, periods=horizon, freq=self.config.freq)

        features = []
        for date in future_dates:
            week = date.isocalendar().week
            month = date.month

            feat = [
                np.sin(2 * np.pi * month / 12),
                np.cos(2 * np.pi * month / 12),
                np.sin(2 * np.pi * week / 52),
                np.cos(2 * np.pi * week / 52),
            ]
            features.append(feat)

        return np.array(features)

    def _ensure_numeric_context(self, df: pd.DataFrame, learn: bool) -> pd.DataFrame:
        """
        Ensure context features are numeric with PERSISTENT category mapping.
        If learn=True, store categories seen during fit.
        If learn=False, apply stored categories and map unknowns to an 'unknown' index.
        """
        df = df.copy()
        try:
            from pandas.api.types import CategoricalDtype
        except Exception:  # pragma: no cover
            CategoricalDtype = None  # type: ignore

        for col in self.config.context_features:
            if col not in df.columns:
                continue
            if np.issubdtype(df[col].dtype, np.number):
                # Already numeric
                continue
            if learn:
                # Learn categories from training data
                df[col] = df[col].astype('category')
                cats = list(df[col].cat.categories)
                self._context_categories[col] = cats  # persist
                df[col] = df[col].cat.codes.astype('int32')
            else:
                # Apply stored categories; map unknowns to last index
                cats = self._context_categories.get(col, [])
                if CategoricalDtype is not None and len(cats) > 0:
                    dtype = CategoricalDtype(categories=cats)
                    df[col] = df[col].astype(dtype)
                    codes = df[col].cat.codes.astype('int32')
                    unknown_idx = len(cats)
                    # pandas uses -1 for unknowns; remap to unknown_idx
                    codes = codes.where(codes != -1, other=unknown_idx)
                    df[col] = codes
                else:
                    # No stored categories; fallback to ordinal encode (best effort)
                    df[col] = df[col].astype('category').cat.codes.astype('int32')
        return df

    def _create_sequences(
        self,
        df: pd.DataFrame,
        is_training: bool = False,
        forecast_start_override: Optional[pd.Timestamp] = None
    ) -> Tuple[np.ndarray, np.ndarray, List[Optional[np.ndarray]], Optional[np.ndarray], List[str], List[pd.Timestamp]]:
        """
        Create sequences with proper handling of mixed few/zero-shot.

        Returns past_sequences as LIST where elements can be None.
        """
        logger.info("Creating sequences...")

        # Ensure numeric context (learn during training; reuse during predict)
        df = self._ensure_numeric_context(df, learn=is_training)

        all_context = []
        all_future_features = []
        all_past_sequences = []
        all_targets = []
        all_item_ids = []
        all_forecast_dates = []

        items_processed = 0
        items_skipped = 0

        for item_id, group in df.groupby(self.config.item_id_col):
            group_sorted = group.sort_values(self.config.date_col).reset_index(drop=True)

            # Extract context
            if self.config.context_features:
                context_values = group_sorted[self.config.context_features].iloc[0].values
            else:
                context_values = np.array([0.0])

            if is_training:
                min_length = self.config.n_steps_lag + self.config.forecast_horizon
                if len(group_sorted) < min_length:
                    items_skipped += 1
                    continue

                for i in range(len(group_sorted) - self.config.n_steps_lag - self.config.forecast_horizon + 1):
                    # Past sequence
                    past_window = group_sorted.iloc[i:i+self.config.n_steps_lag]
                    past_seq = past_window[[self.config.target_col] + self.config.temporal_features].values

                    # Future targets
                    future_window = group_sorted.iloc[
                        i+self.config.n_steps_lag:i+self.config.n_steps_lag+self.config.forecast_horizon
                    ]
                    future_targets = future_window[self.config.target_col].values

                    if len(future_targets) != self.config.forecast_horizon:
                        continue

                    # Future calendar
                    forecast_start = future_window.iloc[0][self.config.date_col]
                    future_feats = self._create_future_features(forecast_start, self.config.forecast_horizon)

                    all_context.append(context_values)
                    all_future_features.append(future_feats)
                    all_past_sequences.append(past_seq)
                    all_targets.append(future_targets)
                    all_item_ids.append(item_id)
                    all_forecast_dates.append(forecast_start)
            else:
                # Prediction mode
                if forecast_start_override is not None:
                    forecast_start = forecast_start_override
                else:
                    last_dt = group_sorted[self.config.date_col].iloc[-1]
                    offset = pd.tseries.frequencies.to_offset(self.config.freq)
                    forecast_start = last_dt + offset

                future_feats = self._create_future_features(forecast_start, self.config.forecast_horizon)

                # Past sequence if available (KEEP AS INDIVIDUAL, NOT ARRAY)
                if len(group_sorted) >= self.config.n_steps_lag:
                    past_window = group_sorted.iloc[-self.config.n_steps_lag:]
                    past_seq = past_window[[self.config.target_col] + self.config.temporal_features].values
                    all_past_sequences.append(past_seq)
                else:
                    # Zero-shot: no history
                    all_past_sequences.append(None)

                all_context.append(context_values)
                all_future_features.append(future_feats)
                all_item_ids.append(item_id)
                all_forecast_dates.append(forecast_start)

            items_processed += 1

        if items_skipped > 0:
            logger.warning(f"Skipped {items_skipped} items with insufficient data")

        if not all_context:
            raise ValidationError("No valid sequences created")

        context = np.array(all_context)
        future_features = np.array(all_future_features)

        # Return past_sequences as LIST (may contain None)
        past_sequences = all_past_sequences

        targets = np.array(all_targets) if all_targets else None

        logger.info(f"Created {len(context)} sequences from {items_processed} items")
        logger.info(f"Context shape: {context.shape}")
        logger.info(f"Future features shape: {future_features.shape}")
        if is_training:
            logger.info(f"Past sequences: {len([p for p in past_sequences if p is not None])} with history")

        return context, future_features, past_sequences, targets, all_item_ids, all_forecast_dates

    def fit(self, df: pd.DataFrame, validation_split: float = 0.2) -> 'MetaFiLMForecaster':
        """Train on a long-format panel.

        Args:
            df: training frame with the columns named in the config.
            validation_split: share of training windows held out for early
                stopping. Scalers are fitted on the remaining windows only.

        Returns:
            The fitted forecaster (``self``).
        """
        logger.info("Starting MetaFiLM training...")

        required_cols = [
            self.config.item_id_col,
            self.config.target_col,
            self.config.date_col
        ] + self.config.context_features + self.config.temporal_features

        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValidationError(f"Missing columns: {missing}")

        # Create sequences
        context, future_features, past_sequences_list, targets, _, _ = self._create_sequences(
            df, is_training=True
        )

        # Convert past_sequences list to array (all have data in training)
        past_sequences = np.array(past_sequences_list)

        # Handle NaNs before scaling (prevent propagation)
        logger.info("Handling NaNs in features...")
        context = np.nan_to_num(context, nan=0.0)
        targets = np.nan_to_num(targets, nan=0.0)
        future_features = np.nan_to_num(future_features, nan=0.0)
        past_sequences = np.nan_to_num(past_sequences, nan=0.0)

        # SPLIT FIRST
        n = len(context)
        n_val = int(n * validation_split)
        indices = np.random.permutation(n)
        train_idx = indices[n_val:]
        val_idx = indices[:n_val]

        # FIT SCALERS ON TRAIN ONLY
        self.context_scaler_ = StandardScaler().fit(context[train_idx])

        # Flatten target for single scale
        self.target_scaler_ = StandardScaler().fit(targets[train_idx].reshape(-1, 1))

        # Future features
        future_train_flat = future_features[train_idx].reshape(-1, future_features.shape[-1])
        self.future_scaler_ = StandardScaler().fit(future_train_flat)

        # Past sequences
        past_train_flat = past_sequences[train_idx].reshape(-1, past_sequences.shape[-1])
        self.past_scaler_ = StandardScaler().fit(past_train_flat)

        # TRANSFORM SEPARATELY
        def transform_split(idx):
            ctx = self.context_scaler_.transform(context[idx])

            tgt = self.target_scaler_.transform(targets[idx].reshape(-1, 1))
            tgt = tgt.reshape(len(idx), -1)

            fut = future_features[idx].reshape(-1, future_features.shape[-1])
            fut = self.future_scaler_.transform(fut)
            fut = fut.reshape(len(idx), *future_features.shape[1:])

            pst = past_sequences[idx].reshape(-1, past_sequences.shape[-1])
            pst = self.past_scaler_.transform(pst)
            pst = pst.reshape(len(idx), *past_sequences.shape[1:])

            # Convert to list for dataset
            pst_list = [pst[i] for i in range(len(idx))]

            return ctx, fut, pst_list, tgt

        ctx_train, fut_train, pst_train, tgt_train = transform_split(train_idx)
        ctx_val, fut_val, pst_val, tgt_val = transform_split(val_idx)

        # Create datasets
        train_dataset = TimeSeriesDataset(ctx_train, fut_train, pst_train, tgt_train)
        val_dataset = TimeSeriesDataset(ctx_val, fut_val, pst_val, tgt_val)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=_collate_train
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=_collate_train
        )

        # Initialize model
        past_dim = past_sequences.shape[-1]
        future_dim = future_features.shape[-1]
        context_dim = context.shape[-1]

        self.model_ = MetaFiLMModel(
            past_feature_dim=past_dim,
            future_feature_dim=future_dim,
            context_dim=context_dim,
            hidden_dim=self.config.hidden_dim,
            num_lstm_layers=self.config.num_lstm_layers,
            forecast_horizon=self.config.forecast_horizon,
            dropout=self.config.dropout,
            use_residual=self.config.use_residual,
            film_mode=self.config.film_mode,
            film_init=self.config.film_init,
            use_level=self.config.use_level,
            use_history=self.config.use_history
        ).to(self.config.device)

        # Optimizer with weight decay only on the metadata-fusion module.  The
        # reviewer-requested concat arm uses the same optimisation treatment as
        # FiLM so the comparison changes the fusion operation, not regularisation.
        fusion_module = (
            self.model_.film_layer.concat_fusion
            if self.config.film_mode == "concat"
            else self.model_.film_layer.film_generator
        )
        fusion_params = list(fusion_module.parameters())
        fusion_param_ids = {id(p) for p in fusion_params}
        other_params = [p for p in self.model_.parameters() if id(p) not in fusion_param_ids]

        optimizer = optim.Adam([
            {'params': other_params, 'weight_decay': 0.0},
            {'params': fusion_params, 'weight_decay': self.config.weight_decay}
        ], lr=self.config.learning_rate)

        criterion = {
            "mse": nn.MSELoss,
            "l1": nn.L1Loss,
            "huber": nn.HuberLoss,
        }[self.config.loss]()

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.config.num_epochs):
            # Train
            self.model_.train()
            train_loss = 0.0

            for batch in train_loader:
                ctx, fut, pst, tgt = batch

                ctx = ctx.to(self.config.device)
                fut = fut.to(self.config.device)
                pst = pst.to(self.config.device) if pst is not None else None
                tgt = tgt.to(self.config.device)

                optimizer.zero_grad()
                pred = self.model_(ctx, fut, pst)
                loss = criterion(pred, tgt)
                loss.backward()
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item()

            train_batches = max(1, len(train_loader))
            train_loss /= train_batches
            self.train_losses_.append(train_loss)

            # Validate
            self.model_.eval()
            val_loss = 0.0

            with torch.no_grad():
                for batch in val_loader:
                    ctx, fut, pst, tgt = batch

                    ctx = ctx.to(self.config.device)
                    fut = fut.to(self.config.device)
                    pst = pst.to(self.config.device) if pst is not None else None
                    tgt = tgt.to(self.config.device)

                    pred = self.model_(ctx, fut, pst)
                    loss = criterion(pred, tgt)
                    val_loss += loss.item()

            val_batches = max(1, len(val_loader))
            val_loss /= val_batches
            self.val_losses_.append(val_loss)

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}: Train={train_loss:.4f}, Val={val_loss:.4f}")

            # Early stopping. The snapshot must be a copy: state_dict() returns
            # tensors that share storage with the live parameters.
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                if self.config.restore_best_weights:
                    self.best_model_state_ = {
                        k: v.detach().clone() for k, v in self.model_.state_dict().items()
                    }
            else:
                patience_counter += 1

            if patience_counter >= self.config.patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        if self.config.restore_best_weights and hasattr(self, 'best_model_state_'):
            self.model_.load_state_dict(self.best_model_state_)

        self.is_fitted_ = True
        logger.info("Training completed")

        return self

    def predict(
        self,
        df: pd.DataFrame,
        item_id: Optional[str] = None,
        forecast_start_date: Optional[pd.Timestamp] = None
    ) -> Union[Dict[str, np.ndarray], np.ndarray]:
        """Forecast the next ``forecast_horizon`` periods for every series.

        Series with at least ``n_steps_lag`` rows use their latest history;
        shorter or empty histories are forecast from metadata and calendar
        alone (cold start).

        Args:
            df: frame with the same columns as at fit time. A cold-start
                series needs at least one row carrying its metadata.
            item_id: forecast only this series and return its array.
            forecast_start_date: first forecast period for all series;
                defaults to the period after each series' last row.

        Returns:
            ``{item_id: array of shape (forecast_horizon,)}``, or a single
            array when ``item_id`` is given.
        """
        if not self.is_fitted_:
            raise ModelNotFittedError("Model must be fitted first")

        if item_id is not None:
            df = df[df[self.config.item_id_col] == item_id].copy()

        # Ensure we apply the SAME categorical mapping as fit-time
        df = self._ensure_numeric_context(df, learn=False)

        context, future_features, past_sequences_list, _, item_ids, _ = self._create_sequences(
            df, is_training=False, forecast_start_override=forecast_start_date
        )

        # NaN safety
        context = np.nan_to_num(context, nan=0.0)
        future_features = np.nan_to_num(future_features, nan=0.0)

        # Transform context and future
        ctx_scaled = self.context_scaler_.transform(context)
        fut_scaled = self.future_scaler_.transform(
            future_features.reshape(-1, future_features.shape[-1])
        ).reshape(future_features.shape)

        # Scale past sequences individually (handle None)
        past_list = []
        for pst in past_sequences_list:
            if pst is None:
                past_list.append(None)
            else:
                pst = np.nan_to_num(pst, nan=0.0)
                pst_scaled = self.past_scaler_.transform(pst)
                past_list.append(pst_scaled)

        # Predict per-sample (preserves few-shot where available)
        self.model_.eval()
        predictions = []

        with torch.no_grad():
            for i in range(len(ctx_scaled)):
                ctx = torch.as_tensor(ctx_scaled[i:i+1], dtype=torch.float32, device=self.config.device)
                fut = torch.as_tensor(fut_scaled[i:i+1], dtype=torch.float32, device=self.config.device)

                if past_list[i] is not None:
                    pst = torch.as_tensor(past_list[i][None, ...], dtype=torch.float32, device=self.config.device)
                else:
                    pst = None

                pred = self.model_(ctx, fut, pst)
                predictions.append(pred.cpu().numpy())

        predictions = np.vstack(predictions)

        # Inverse transform with flatten/reshape
        predictions = self.target_scaler_.inverse_transform(
            predictions.reshape(-1, 1)
        ).reshape(predictions.shape)

        if item_id is not None:
            return predictions[0]
        else:
            return {str(iid): pred for iid, pred in zip(item_ids, predictions)}

    def save(self, path: str) -> None:
        """Save weights, config, scalers and learned categories to ``path``."""
        if not self.is_fitted_ or self.model_ is None:
            raise ModelNotFittedError("Model must be fitted before saving")
        payload = {
            'state_dict': self.model_.state_dict(),
            'config': self.config,
            'context_scaler': self.context_scaler_,
            'target_scaler': self.target_scaler_,
            'past_scaler': self.past_scaler_,
            'future_scaler': self.future_scaler_,
            'context_categories': self._context_categories,
        }
        torch.save(payload, path)
        logger.info(f"Saved MetaFiLM forecaster to {path}")

    @classmethod
    def load(cls, path: str, map_location: Optional[str] = 'cpu') -> "MetaFiLMForecaster":
        """Load a forecaster written by :meth:`save`.

        The file is a pickle (config and fitted scalers), so only load files
        you trust.
        """
        payload = torch.load(path, map_location=map_location, weights_only=False)
        config: MetaFiLMConfig = payload['config']
        model = cls(config)
        model._context_categories = payload.get('context_categories', {})
        model.context_scaler_ = payload['context_scaler']
        model.target_scaler_ = payload['target_scaler']
        model.past_scaler_ = payload['past_scaler']
        model.future_scaler_ = payload['future_scaler']
        # Initialize network with correct dims
        past_dim = int(model.past_scaler_.mean_.shape[0])
        future_dim = int(model.future_scaler_.mean_.shape[0])
        context_dim = int(model.context_scaler_.mean_.shape[0])
        model.model_ = MetaFiLMModel(
            past_feature_dim=past_dim,
            future_feature_dim=future_dim,
            context_dim=context_dim,
            hidden_dim=config.hidden_dim,
            num_lstm_layers=config.num_lstm_layers,
            forecast_horizon=config.forecast_horizon,
            dropout=config.dropout,
            use_residual=getattr(config, "use_residual", True),
            film_mode=getattr(config, "film_mode", "bounded"),
            film_init=getattr(config, "film_init", "default"),
            use_level=getattr(config, "use_level", True),
            use_history=getattr(config, "use_history", True)
        ).to(config.device)
        model.model_.load_state_dict(payload['state_dict'])
        model.is_fitted_ = True
        logger.info(f"Loaded MetaFiLM forecaster from {path}")
        return model

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {"config": self.config}

    def set_params(self, **params) -> 'MetaFiLMForecaster':
        if "config" in params:
            self.config = params["config"]
            self.is_fitted_ = False
        return self
