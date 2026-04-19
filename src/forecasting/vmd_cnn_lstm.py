"""
VMD (Variational Mode Decomposition) + CNN-LSTM 수요 예측 모델

절차:
1. 학습 수요 시계열을 VMD로 K개 모드(IMF)로 분해
2. 각 모드의 래그 피처를 기존 피처에 추가하여 X 증강
3. CNN(다중 커널) + BiLSTM + Attention 으로 예측

입력: (batch, seq_len=168, n_features+K)
출력: (batch, horizon=24)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path


# ─────────────────────────────────────────────────────────────
# 1. VMD 구현 (vmdpy 없이 numpy만 사용)
# ─────────────────────────────────────────────────────────────

def vmd(signal: np.ndarray, K: int = 5, alpha: float = 2000.0,
        tau: float = 0.0, tol: float = 1e-7, max_iter: int = 500) -> np.ndarray:
    """
    Variational Mode Decomposition (Dragomiretskiy & Zosso, 2014)

    Parameters
    ----------
    signal : (N,) 수요 시계열
    K      : 분해 모드 수
    alpha  : 대역폭 제약 (클수록 좁은 대역)
    tau    : Lagrangian 업데이트 스텝 (0 = noise-tolerant)
    tol    : 수렴 허용치
    max_iter: 최대 반복

    Returns
    -------
    u : (K, N) — K개 모드 (IMF)
    """
    T = len(signal)

    # ── 미러링으로 경계 효과 제거 ──────────────────────────────
    f_mirror = np.concatenate([signal[T//2-1::-1], signal, signal[-1:T//2-1:-1]])
    T2 = len(f_mirror)

    # 주파수 축 [-0.5, 0.5)
    freqs = (np.arange(T2) / T2) - 0.5

    # 단측 스펙트럼
    f_hat = np.fft.fftshift(np.fft.fft(f_mirror))
    f_hat_plus = f_hat.copy()
    f_hat_plus[:T2 // 2] = 0.0

    # ── 초기화 ────────────────────────────────────────────────
    u_hat = np.zeros((T2, K), dtype=complex)          # 각 모드 스펙트럼
    u_hat_prev = u_hat.copy()
    omega = np.array([(0.5 / K) * k for k in range(K)])   # 중심 주파수
    lam = np.zeros(T2, dtype=complex)                 # 이중 변수

    half = T2 // 2

    # ── ADMM 반복 ─────────────────────────────────────────────
    for n in range(max_iter):
        sum_uk = u_hat.sum(axis=1)

        for k in range(K):
            # 다른 모드 합
            sum_uk = sum_uk - u_hat[:, k]

            # u_k 업데이트
            denom = 1.0 + alpha * (freqs - omega[k]) ** 2
            u_hat[:, k] = (f_hat_plus - sum_uk - lam / 2.0) / denom
            sum_uk = sum_uk + u_hat[:, k]

            # omega_k 업데이트 (단측 스펙트럼 가중 평균)
            power = np.abs(u_hat[half:, k]) ** 2
            omega[k] = np.dot(freqs[half:], power) / (power.sum() + 1e-12)

        # Lagrange 업데이트
        lam += tau * (sum_uk - f_hat_plus)

        # 수렴 확인
        diff = np.sum(np.abs(u_hat - u_hat_prev) ** 2) / (np.sum(np.abs(u_hat_prev) ** 2) + 1e-12)
        u_hat_prev = u_hat.copy()
        if diff < tol:
            break

    # ── 시간 영역 복원 ────────────────────────────────────────
    u_full = np.zeros((K, T2))
    for k in range(K):
        spec = u_hat[:, k] + np.conj(np.flipud(u_hat[:, k]))
        u_full[k] = np.real(np.fft.ifft(np.fft.ifftshift(spec)))

    # 미러 제거: 원본 구간
    start = T2 // 4
    u = u_full[:, start: start + T]

    # 에너지 순으로 정렬 (낮은 주파수→높은 주파수)
    energy = (u ** 2).sum(axis=1)
    u = u[np.argsort(-energy)]

    return u.astype(np.float32)   # (K, T)


# ─────────────────────────────────────────────────────────────
# 2. VMD 피처 증강
# ─────────────────────────────────────────────────────────────

def augment_with_vmd(
    X_train: np.ndarray,
    X_val:   np.ndarray,
    X_test:  np.ndarray,
    demand_col_idx: int = 0,   # X 내 수요 컬럼 인덱스 (demand_lag24)
    K: int = 5,
    alpha: float = 2000.0,
) -> tuple:
    """
    학습셋 전체 수요 시계열에 VMD를 적용해 K개 모드를 피처로 추가.

    Returns
    -------
    X_train_aug, X_val_aug, X_test_aug : (N, seq_len, n_features+K)
    """
    print(f"VMD 분해 중 (K={K}, alpha={alpha})...")

    # ── 학습셋 수요 컬럼으로 전체 시계열 재구성 ──────────────
    # X_train[i, t, demand_col_idx] = i+t 번째 수요 (슬라이딩 윈도우)
    # 첫 번째 샘플의 모든 타임스텝 + 이후 샘플들의 마지막 스텝
    demand_train = X_train[:, :, demand_col_idx]   # (N, seq_len)
    # 첫 윈도우 전체 + 이후 마지막 한 스텝씩
    full_series = demand_train[0].tolist()
    for i in range(1, len(demand_train)):
        full_series.append(float(demand_train[i, -1]))
    full_series = np.array(full_series, dtype=np.float32)

    # ── VMD 분해 ────────────────────────────────────────────
    modes = vmd(full_series, K=K, alpha=alpha)    # (K, len(full_series))

    # ── 각 세트에 모드 피처 추가 ────────────────────────────
    def add_modes(X, modes_arr, n_full_train):
        N, seq_len, n_feat = X.shape
        aug = np.zeros((N, seq_len, n_feat + K), dtype=np.float32)
        aug[:, :, :n_feat] = X

        for i in range(N):
            # 학습셋: 직접 인덱싱 가능
            # Val/Test: 학습셋 이후 구간 → 모드 값 없음 → 0으로 채움
            # (더 정확하게 하려면 전체 시계열을 VMD 해야 하지만, 학습셋만으로 근사)
            start = i
            end   = i + seq_len
            if end <= modes_arr.shape[1]:
                for k in range(K):
                    aug[i, :, n_feat + k] = modes_arr[k, start:end]
        return aug

    X_train_aug = add_modes(X_train, modes, len(full_series))
    # Val/Test는 학습셋 모드 범위 밖 → 0 패딩 (추론 시 모드 재계산 필요)
    X_val_aug   = np.concatenate([X_val,  np.zeros((len(X_val),  X_val.shape[1],  K), dtype=np.float32)], axis=2)
    X_test_aug  = np.concatenate([X_test, np.zeros((len(X_test), X_test.shape[1], K), dtype=np.float32)], axis=2)

    print(f"피처 확장: {X_train.shape[-1]} → {X_train_aug.shape[-1]}")
    return X_train_aug, X_val_aug, X_test_aug


# ─────────────────────────────────────────────────────────────
# 3. CNN-LSTM 모델
# ─────────────────────────────────────────────────────────────

class MultiScaleCNN(nn.Module):
    """다중 커널 1D CNN — 시간적 패턴을 다양한 스케일로 추출"""
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_sizes=(3, 7, 14, 24)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, k, padding=k // 2),
                nn.BatchNorm1d(out_channels),
                nn.GELU(),
            ) for k in kernel_sizes
        ])

    def forward(self, x):
        # x: (B, C, T)  — 짝수 kernel은 출력이 T+1이 될 수 있으므로 트리밍
        T = x.shape[-1]
        outs = [branch(x)[..., :T] for branch in self.branches]
        return torch.cat(outs, dim=1)   # (B, out_channels × n_kernels, T)


class Attention(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.attn = nn.Linear(hidden * 2, 1)

    def forward(self, lstm_out):
        w = torch.softmax(self.attn(lstm_out), dim=1)
        return (w * lstm_out).sum(dim=1)


class VMDCNNLSTMModel(nn.Module):
    """
    VMD + CNN-LSTM:
    Input → MultiScaleCNN → BiLSTM → Attention → FC → horizon
    """
    def __init__(
        self,
        input_size:  int,
        cnn_channels: int = 32,
        kernel_sizes: tuple = (3, 7, 14, 24),
        lstm_hidden:  int = 256,
        lstm_layers:  int = 2,
        horizon:      int = 24,
        dropout:      float = 0.2,
    ):
        super().__init__()
        n_kernels = len(kernel_sizes)
        cnn_out   = cnn_channels * n_kernels   # concat 후 채널 수

        self.cnn  = MultiScaleCNN(input_size, cnn_channels, kernel_sizes)
        self.lstm = nn.LSTM(
            cnn_out, lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.norm    = nn.LayerNorm(lstm_hidden * 2)
        self.attn    = Attention(lstm_hidden)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, horizon),
        )

    def forward(self, x):
        # x: (B, T, F)
        h = x.permute(0, 2, 1)          # (B, F, T) — Conv1d 형식
        h = self.cnn(h)                  # (B, cnn_out, T)
        h = h.permute(0, 2, 1)          # (B, T, cnn_out)
        h, _ = self.lstm(h)             # (B, T, hidden*2)
        h = self.norm(h)
        ctx = self.attn(h)              # (B, hidden*2)
        ctx = self.dropout(ctx)
        return self.fc(ctx), None       # (B, horizon)


# ─────────────────────────────────────────────────────────────
# 4. 학습 함수
# ─────────────────────────────────────────────────────────────

def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    save_dir:     str   = "models",
    save_name:    str   = None,
    cnn_channels: int   = 32,
    kernel_sizes: tuple = (3, 7, 14, 24),
    lstm_hidden:  int   = 256,
    lstm_layers:  int   = 2,
    epochs:       int   = 300,
    batch_size:   int   = 128,
    lr:           float = 5e-4,
    patience:     int   = 20,
    device:       str   = None,
) -> VMDCNNLSTMModel:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Model: VMD-CNN-LSTM | Device: {device}")

    def to_loader(X, y, shuffle):
        ds = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          pin_memory=(device == "cuda"), num_workers=0)

    train_loader = to_loader(X_train, y_train, True)
    val_loader   = to_loader(X_val,   y_val,   False)

    model = VMDCNNLSTMModel(
        input_size=X_train.shape[2],
        cnn_channels=cnn_channels,
        kernel_sizes=kernel_sizes,
        lstm_hidden=lstm_hidden,
        lstm_layers=lstm_layers,
        horizon=y_train.shape[1],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5, min_lr=1e-6
    )
    criterion = nn.HuberLoss()

    _save_name = save_name or "demand_vmd_cnn_lstm_best.pt"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    best_val   = float("inf")
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred, _ = model(Xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(Xb)
        train_loss /= len(X_train)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                pred, _ = model(Xb)
                val_loss += criterion(pred, yb).item() * len(Xb)
        val_loss /= len(X_val)

        scheduler.step(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | train: {train_loss:.4f} | val: {val_loss:.4f}")

        if val_loss < best_val:
            best_val   = val_loss
            no_improve = 0
            torch.save(model.state_dict(), f"{save_dir}/{_save_name}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"Best val loss: {best_val:.4f}")
    model.load_state_dict(torch.load(f"{save_dir}/{_save_name}", map_location=device))
    return model


# ─────────────────────────────────────────────────────────────
# 5. 추론
# ─────────────────────────────────────────────────────────────

def predict(model: VMDCNNLSTMModel, X: np.ndarray,
            device: str = None, batch_size: int = 512) -> np.ndarray:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model.eval().to(device)
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            Xb = torch.FloatTensor(X[i:i + batch_size]).to(device)
            p, _ = model(Xb)
            preds.append(p.cpu().numpy())
    return np.concatenate(preds, axis=0)


def load_model(path: str, input_size: int, lstm_hidden: int = 256,
               lstm_layers: int = 2, horizon: int = 24) -> VMDCNNLSTMModel:
    model = VMDCNNLSTMModel(input_size, lstm_hidden=lstm_hidden,
                            lstm_layers=lstm_layers, horizon=horizon)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model
