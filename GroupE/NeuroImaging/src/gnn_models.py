# gnn_models.py

from config import opt

if opt['gnn']:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset
    from torch_geometric.nn import GCNConv, global_mean_pool

    # ---------------- Dataset for forecasting (used by DAST)
    class WindowForecastDS(Dataset):
        def __init__(self, tensor, seq_len):
            self.tensor = tensor
            self.seq_len = seq_len
        def __len__(self):
            return len(self.tensor) - self.seq_len
        def __getitem__(self, idx):
            x = self.tensor[idx : idx + self.seq_len]
            y = self.tensor[idx + self.seq_len]
            return torch.from_numpy(x), torch.from_numpy(y)

    # ---------------- BrainGCN (graph-based classifier)
    class BrainGCN(nn.Module):
        def __init__(self, in_feats=1, h_feats=16, n_classes=2):
            super().__init__()
            self.conv1 = GCNConv(in_feats, h_feats)
            self.conv2 = GCNConv(h_feats, h_feats)
            self.lin   = nn.Linear(h_feats, n_classes)
        def forward(self, x, edge_index, batch):
            x = self.conv1(x, edge_index).relu()
            x = self.conv2(x, edge_index).relu()
            x = global_mean_pool(x, batch)
            return self.lin(x)

    # ---------------- MLP baseline (for raw/resid FC data)
    class MLPBaseline(nn.Module):
        def __init__(self, input_size, hidden_size=64, num_classes=2):
            super().__init__()
            self.model = nn.Sequential(
                nn.Flatten(),  # Converts (B, N, N) → (B, N*N)
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, num_classes)
            )
        def forward(self, x):
            return self.model(x)
    class MLPBaselineImproved(nn.Module):
        def __init__(self, input_size, num_classes=2, hidden_sizes=[128, 64], dropout=0.15):
            super().__init__()
            self.flatten = nn.Flatten()

            self.fc1 = nn.Linear(input_size, hidden_sizes[0])
            self.bn1 = nn.BatchNorm1d(hidden_sizes[0])
            self.act1 = nn.GELU()   # یا LeakyReLU، GELU معمولاً بهتره

            self.dropout1 = nn.Dropout(dropout)

            self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
            self.bn2 = nn.BatchNorm1d(hidden_sizes[1])
            self.act2 = nn.GELU()
            self.dropout2 = nn.Dropout(dropout)

            self.out = nn.Linear(hidden_sizes[1], num_classes)

        def forward(self, x):
            x = self.flatten(x)
            x = self.fc1(x)
            x = self.bn1(x)
            x = self.act1(x)
            x = self.dropout1(x)
            x = self.fc2(x)
            x = self.bn2(x)
            x = self.act2(x)
            x = self.dropout2(x)
            x = self.out(x)
            return x

    # ---------------- DAST Forecaster (for dFC time series)
    class DASTBlock(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.temp = nn.Conv2d(d, d, (3,1), padding=(1,0), groups=d)
            self.ag   = nn.Linear(d, d)
            self.act  = nn.ReLU()
        def forward(self, x):
            b, t, n, d = x.shape
            x = x.permute(0, 3, 1, 2)           # (B, D, T, N)
            x = self.temp(x).permute(0, 2, 3, 1) # (B, T, N, D)
            x = self.ag(x)
            return self.act(x)

    class DASTForecaster(nn.Module):
        def __init__(self, n_roi, d=64, k=2):
            super().__init__()
            self.n_roi = n_roi
            self.inp = nn.Linear(n_roi, d)
            self.blks = nn.ModuleList([DASTBlock(d) for _ in range(k)])
            self.out = nn.Linear(d, n_roi)
        def forward(self, x):
            assert x.dim() == 4, f"Input must be 4D tensor (B,T,N,N), got {x.shape}"
            b, t, n, n2 = x.shape
            assert n == self.n_roi and n2 == self.n_roi, f"Input spatial dims must be n_roi ({self.n_roi}), got ({n}, {n2})"
            x = (x - x.mean()) / x.std()
            x = torch.tanh(x)
            x = x.view(b * t, n, n)
            x = self.inp(x)
            x = x.view(b, t, n, -1)
            for blk in self.blks:
                x = blk(x)
            x = x.mean(dim=1)
            x = self.out(x)
            x = x.permute(0, 2, 1)  # (B, N, T)
            return x
