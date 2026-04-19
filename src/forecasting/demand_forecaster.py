"""
수요 예측 모델
- DemandLSTM: Bidirectional LSTM + Attention
- DemandTFT:  Simplified Temporal Fusion Transformer
입력: (batch, seq_len=168, features) → 출력: (batch, horizon=24)
"""

import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

# FEATURE_COLS 내 hour_sin / hour_cos 인덱스 (preprocessor.py 기준)
_HOUR_SIN_IDX = 13
_HOUR_COS_IDX = 14

# 한국 전력 피크 시간대
_PEAK_HOURS = {9, 10, 11, 18, 19, 20, 21}


# ────────────────────────────────────────────
# 피크 가중 Huber Loss
# ────────────────────────────────────────────

class PeakWeightedHuberLoss(nn.Module):
    """
    배치 내 각 시퀀스의 실제 시각을 X의 hour_sin/hour_cos 로 디코딩하고,
    피크 시간대(9-11시, 18-21시) 출력 스텝에 높은 가중치를 부여합니다.
    """

    def __init__(self, peak_weight: float = 2.5, delta: float = 1.0):
        super().__init__()
        self.peak_weight = peak_weight
        self.delta       = delta

    def forward(self, X: torch.Tensor, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        X      : (B, seq_len, features)
        pred   : (B, horizon=24)
        target : (B, horizon=24)
        """
        B, H = pred.shape
        device = pred.device

        # 마지막 입력 스텝의 hour_sin / hour_cos 로 시작 시각 디코딩
        h_sin  = X[:, -1, _HOUR_SIN_IDX]                                  # (B,)
        h_cos  = X[:, -1, _HOUR_COS_IDX]
        h_prev = torch.round(torch.atan2(h_sin, h_cos) * 24.0 / (2.0 * math.pi)).long() % 24
        h_start = (h_prev + 1) % 24                                        # 예측 시작 시각

        # 각 출력 스텝의 실제 시각: (B, H)
        step_idx     = torch.arange(H, device=device).unsqueeze(0)         # (1, H)
        actual_hours = (h_start.unsqueeze(1) + step_idx) % 24              # (B, H)

        # 피크 마스크 → 가중치
        peak_mask = torch.zeros(B, H, device=device)
        for ph in _PEAK_HOURS:
            peak_mask += (actual_hours == ph).float()
        weights = 1.0 + (self.peak_weight - 1.0) * peak_mask              # 1.0 or peak_weight

        # Element-wise Huber
        diff  = torch.abs(pred - target)
        huber = torch.where(
            diff < self.delta,
            0.5 * diff ** 2,
            self.delta * (diff - 0.5 * self.delta),
        )

        return (huber * weights).mean()


# ────────────────────────────────────────────
# 모델 정의
# ────────────────────────────────────────────

class Attention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Linear(hidden_size * 2, 1)

    def forward(self, lstm_out):
        # lstm_out: (batch, seq_len, hidden*2)
        weights = torch.softmax(self.attn(lstm_out), dim=1)  # (batch, seq_len, 1)
        context = (weights * lstm_out).sum(dim=1)            # (batch, hidden*2)
        return context, weights.squeeze(-1)


class DemandLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 3,
        horizon: int = 24,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention  = Attention(hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.dropout    = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, horizon),
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        out, _ = self.lstm(x)                   # (batch, seq_len, hidden*2)
        context, attn_w = self.attention(out)
        context = self.layer_norm(context)
        context = self.dropout(context)
        pred = self.fc(context)                 # (batch, horizon)
        return pred, attn_w


# ────────────────────────────────────────────
# TFT (Simplified Temporal Fusion Transformer)
# ────────────────────────────────────────────

class GRN(nn.Module):
    """Gated Residual Network"""
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.fc1  = nn.Linear(d_model, d_model)
        self.fc2  = nn.Linear(d_model, d_model)
        self.gate = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        h = self.drop(h)
        h = self.fc2(h)
        g = torch.sigmoid(self.gate(x))
        return self.norm(x + g * h)


class DemandTFT(nn.Module):
    """
    Simplified TFT:
    1. Variable Selection (Linear projection per feature → weighted sum)
    2. LSTM encoder (past context)
    3. Multi-head Self-Attention (long-range dependency)
    4. GRN + FC → horizon
    """
    def __init__(
        self,
        input_size: int,
        d_model:    int = 128,
        n_heads:    int = 8,
        num_lstm_layers: int = 2,
        horizon:    int = 24,
        dropout:    float = 0.1,
        seq_len:    int = 168,
    ):
        super().__init__()
        self.d_model = d_model

        # 1. Feature projection
        self.input_proj = nn.Linear(input_size, d_model)

        # 2. LSTM encoder
        self.lstm = nn.LSTM(
            d_model, d_model,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
        )
        self.lstm_norm = nn.LayerNorm(d_model)

        # 3. Multi-head Attention
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(d_model)

        # 4. GRN + output
        self.grn = GRN(d_model, dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, horizon),
        )

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        h = self.input_proj(x)               # (B, T, d_model)

        # LSTM
        h, _ = self.lstm(h)                  # (B, T, d_model)
        h = self.lstm_norm(h)

        # Self-attention
        h2, _ = self.attn(h, h, h)          # (B, T, d_model)
        h = self.attn_norm(h + h2)

        # Pool last step + GRN
        ctx = h[:, -1, :]                    # (B, d_model)
        ctx = self.grn(ctx)
        ctx = self.dropout(ctx)
        pred = self.fc(ctx)                  # (B, horizon)
        return pred, None


# ────────────────────────────────────────────
# 학습 (공통)
# ────────────────────────────────────────────

def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    save_dir:    str   = "models",
    model_type:  str   = "lstm",    # "lstm" or "tft"
    hidden_size: int   = 256,
    num_layers:  int   = 3,
    epochs:      int   = 200,
    batch_size:  int   = 128,
    lr:          float = 5e-4,
    patience:    int   = 20,
    device:      str   = None,
    save_name:   str   = None,      # None이면 model_type에 따라 자동 결정
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Model: {model_type.upper()} | Device: {device}")

    # DataLoader
    def to_loader(X, y, shuffle):
        ds = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          pin_memory=(device == "cuda"), num_workers=0)

    train_loader = to_loader(X_train, y_train, shuffle=True)
    val_loader   = to_loader(X_val,   y_val,   shuffle=False)

    input_size = X_train.shape[2]
    horizon    = y_train.shape[1]

    if model_type == "tft":
        model = DemandTFT(
            input_size=input_size,
            d_model=hidden_size,
            n_heads=8,
            num_lstm_layers=num_layers,
            horizon=horizon,
        ).to(device)
        _save_name = save_name or "demand_tft_best.pt"
    else:
        model = DemandLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            horizon=horizon,
        ).to(device)
        _save_name = save_name or "demand_lstm_best.pt"

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5, min_lr=1e-6
    )
    criterion = nn.HuberLoss()

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    no_improve    = 0

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            pred, _ = model(X_b)
            loss = criterion(pred, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(X_b)
        train_loss /= len(X_train)

        # Validation (단순 Huber로 측정 — 스케줄러/얼리스탑 기준)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                pred, _ = model(X_b)
                val_loss += criterion(pred, y_b).item() * len(X_b)
        val_loss /= len(X_val)

        scheduler.step(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | train: {train_loss:.4f} | val: {val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve    = 0
            torch.save(model.state_dict(), f"{save_dir}/{_save_name}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"Best val loss: {best_val_loss:.4f}")
    model.load_state_dict(torch.load(f"{save_dir}/{_save_name}", map_location=device))
    return model


# ────────────────────────────────────────────
# 추론
# ────────────────────────────────────────────

def predict(
    model: DemandLSTM,
    X: np.ndarray,
    device: str = None,
    batch_size: int = 512,
) -> np.ndarray:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    model.eval().to(device)
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            X_b = torch.FloatTensor(X[i:i + batch_size]).to(device)
            p, _ = model(X_b)
            preds.append(p.cpu().numpy())
    return np.concatenate(preds, axis=0)


def load_model(path: str, input_size: int, hidden_size: int = 256,
               num_layers: int = 3, horizon: int = 24,
               model_type: str = "lstm"):
    if model_type == "tft":
        model = DemandTFT(input_size, d_model=hidden_size,
                          num_lstm_layers=num_layers, horizon=horizon)
    else:
        model = DemandLSTM(input_size, hidden_size, num_layers, horizon)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model.load_state_dict(torch.load(path, map_location=device))
    return model


# ────────────────────────────────────────────
# 평가 지표
# ────────────────────────────────────────────

def evaluate(y_true: np.ndarray, y_pred: np.ndarray, scaler=None) -> dict:
    """역정규화 후 MAPE, RMSE, MAE 계산"""
    if scaler is not None:
        shape = y_true.shape
        y_true = scaler.inverse_transform(y_true.reshape(-1, 1)).reshape(shape)
        y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).reshape(shape)

    mae  = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE(%)": mape}


if __name__ == "__main__":
    import sys, yaml
    sys.path.insert(0, ".")

    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    proc = cfg["data"]["processed_dir"]
    X_train = np.load(f"{proc}/X_train.npy")
    y_train = np.load(f"{proc}/y_train.npy")
    X_val   = np.load(f"{proc}/X_val.npy")
    y_val   = np.load(f"{proc}/y_val.npy")

    model = train(X_train, y_train, X_val, y_val, save_dir="models")

    X_test = np.load(f"{proc}/X_test.npy")
    y_test = np.load(f"{proc}/y_test.npy")
    y_pred = predict(model, X_test)
    metrics = evaluate(y_test, y_pred)
    print("Test 성능:", metrics)
