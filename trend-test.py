# =========================
# 1) ライブラリ
# =========================
# pip install pytrends torch matplotlib pandas scikit-learn
from pytrends.request import TrendReq
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# =========================
# 2) Googleトレンドからデータ取得
# =========================
KEYWORD = "Bitcoin"          # ←好きな単語に変更（日本語OK）
GEO = "JP"                   # 地域（"JP","US",""=全世界 など）
TIMEFRAME = "today 5-y"      # 期間（例: "today 12-m", "today 5-y", "2019-01-01 2025-10-01"）

pytrends = TrendReq(hl='ja-JP', tz=540)  # 日本時間
pytrends.build_payload([KEYWORD], timeframe=TIMEFRAME, geo=GEO)
df_trend = pytrends.interest_over_time()

# 取得結果の整形
if df_trend.empty:
    raise ValueError("Googleトレンドからデータが取得できませんでした。キーワード/期間/地域を見直してください。")

# isPartial列を削除して、欠損を前方埋め
series = df_trend[KEYWORD].copy()
series = series.asfreq(series.index.inferred_freq) if series.index.inferred_freq else series
series = series.fillna(method="ffill")
series.name = "value"

# =========================
# 3) 前処理（標準化＆学習/テスト分割）
# =========================
# 学習80% / テスト20%
n = len(series)
train_size = int(n * 0.8)
train_idx_end = train_size

train_series = series.iloc[:train_idx_end]
test_series  = series.iloc[train_idx_end:]

# 標準化（学習データでfit → 両方にtransform）
scaler = StandardScaler()
train_vals = scaler.fit_transform(train_series.values.reshape(-1, 1)).flatten()
test_vals  = scaler.transform(test_series.values.reshape(-1, 1)).flatten()

# =========================
# 4) PyTorch Dataset
# =========================
SEQ_LEN = 24  # 1ステップ予測用の履歴長（週次なら~半年、日次なら約1ヶ月）

#### 以前に使ったTimeSeriesDataset(Dataset)との違いに注目(やっていることはほぼ同じだけど、もし後で拡張しようとすると)
class SeqDataset(Dataset):
    def __init__(self, arr, seq_len=24):
        self.x = []
        self.y = []
        for i in range(len(arr) - seq_len):
            self.x.append(arr[i:i+seq_len])
            self.y.append(arr[i+seq_len])
        self.x = torch.tensor(np.array(self.x), dtype=torch.float32).unsqueeze(-1) # (N, seq_len, 1)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32)               # (N,)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

train_ds = SeqDataset(train_vals, SEQ_LEN)
test_ds  = SeqDataset(test_vals,  SEQ_LEN)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

##ここは以前と同じ
class LSTMModel(nn.Module): 
    def __init__(self, input_size=1, hidden_size=50, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True ###ここに注意
                            )
        self.fc = nn.Linear(hidden_size, 1)  ###ここの意味は？
    def forward(self, x):
        out, _ = self.lstm(x)  # (B, T, H)
        out = out[:, -1, :]   # 最後の時点だけ出力する
        out = self.fc(out)   # (B, 1)
        return out.squeeze() # (B,)

    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMModel(hidden_size=64, num_layers=1).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# =========================
# 6) 学習
# =========================
EPOCHS = 15
model.train()
for epoch in range(EPOCHS):
    epoch_loss = 0.0
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * x.size(0)

    print(f"Epoch {epoch+1}/{EPOCHS}  loss={epoch_loss/len(train_ds):.4f}")

# =========================
# 7) テストデータで 1ステップ予測
# =========================
model.eval()
pred_scaled = []
with torch.no_grad():
    for i in range(len(test_ds)):
        x, _ = test_ds[i]
        x = x.unsqueeze(0).to(device)  # (1, T, 1)
        yhat = model(x).cpu().item()
        pred_scaled.append(yhat)

# 逆標準化（スカラーに対して inverse_transform を使えるように2次元化）
pred_scaled_arr = np.array(pred_scaled).reshape(-1, 1)
pred_inv = scaler.inverse_transform(pred_scaled_arr).flatten()

# 真値（テスト部分のうち、SEQ_LEN 以降が予測対象）
true_inv = test_series.values[SEQ_LEN:]

# 対応するインデックス
plot_index = test_series.index[SEQ_LEN:]

# =========================
# 8) プロット（真値 vs 予測）
# =========================
plt.figure(figsize=(12, 6))
plt.plot(plot_index, true_inv, label="True")
plt.plot(plot_index, pred_inv, label="Predicted")
plt.title(f"Google Trends: {KEYWORD}  — LSTM Prediction vs True")
plt.xlabel("Date")
plt.ylabel("Trend index (0–100)")
plt.legend()
plt.tight_layout()
plt.show()

# =========================
# 9)（おまけ）数ステップ先の再帰予測（テスト末尾から先へ）
# =========================
# 直近の実データ（学習+テストの最後のSEQ_LEN点）を使って、kステップ先まで再帰的に予測
K_STEPS = 12
hist_full = scaler.transform(series.values.reshape(-1,1)).flatten()
window = hist_full[-SEQ_LEN:].copy()

future_scaled = []
model.eval()
with torch.no_grad():
    for _ in range(K_STEPS):
        xin = torch.tensor(window, dtype=torch.float32).view(1, SEQ_LEN, 1).to(device)
        yhat = model(xin).cpu().item()
        future_scaled.append(yhat)
        # 窓を1つ進める
        window = np.concatenate([window[1:], [yhat]])

future = scaler.inverse_transform(np.array(future_scaled).reshape(-1,1)).flatten()
future_index = pd.date_range(series.index[-1], periods=K_STEPS+1, freq=series.index.inferred_freq)[1:]

plt.figure(figsize=(12, 4))
plt.plot(series.index[-100:], series.values[-100:], label="History (last 100)")
plt.plot(future_index, future, label=f"Recursive forecast (+{K_STEPS})")
plt.title(f"Google Trends: {KEYWORD} — Recursive Forecast")
plt.xlabel("Date")
plt.ylabel("Trend index (0–100)")
plt.legend()
plt.tight_layout()
plt.show()
