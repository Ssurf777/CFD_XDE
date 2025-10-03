# main.py — Simple training (no callbacks), PyTorch backend, multi-npy support
import os
import glob
import json
import numpy as np
import deepxde as dde

# ---- Force PyTorch backend ----
dde.backend.set_default_backend("pytorch")
import torch
import torch.nn as nn
import torch.nn.functional as F



# ---- Local utilities ----
from utils.user_define_function import (
    process_pipe_data,
    min_max_scale,
    PointCloudGeometry,
    continuity_pde_noslip,
)

def load_cfd_dataset():
    """
    Returns:
        data: np.ndarray of shape (N, 8) with columns [x,y,z,wd,u,v,w,p]
    """
    # 1) If data_npy/ exists, stack all *.npy
    data_dir = "data_npy"
    if os.path.isdir(data_dir):
        file_list = sorted(glob.glob(os.path.join(data_dir, "*.npy")))
        if not file_list:
            raise FileNotFoundError(f"No .npy files found in directory: {data_dir}")
        all_np = []
        for f in file_list:
            print(f"[INFO] Loading: {f}")
            # process_pipe_data returns many arrays; the last is combined ndarray with columns [x,y,z,wd,u,v,w,p]
            *_rest, data_np = process_pipe_data(f)
            all_np.append(data_np)
        data = np.vstack(all_np)
        print(f"[INFO] Stacked data shape: {data.shape}")
        return data

    # 2) Fallback: single file cfd_data.npy in CWD
    single_path = "cfd_data.npy"
    if os.path.exists(single_path):
        print(f"[INFO] Loading single file: {single_path}")
        *_rest, data_np = process_pipe_data(single_path)
        print(f"[INFO] Data shape: {data_np.shape}")
        return data_np

    raise FileNotFoundError("No data found. Put .npy files into ./data_npy or place ./cfd_data.npy.")
# --- pointwise conv. ---    
class PointNetConvBlock(nn.Module):
    """Pointwise Conv (1x1) + SiLU"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.act = nn.SiLU()

    def forward(self, x):
        # x: (B, in_channels, N)   ← N点を扱う
        return self.act(self.conv(x))  # (B, out_channels, N)


class PointNetLike(nn.Module):
    def __init__(self, in_dim=4, out_dim=4):
        super().__init__()
        # Pointwise conv layers
        self.feat = nn.Sequential(
            PointNetConvBlock(in_dim, 64),
            PointNetConvBlock(64, 128),
            PointNetConvBlock(128, 128),
        )
        # Head MLP
        self.head = nn.Sequential(
            nn.Linear(128*2, 256), nn.SiLU(),
            nn.Linear(256, 128), nn.SiLU(),
            nn.Linear(128, out_dim),
        )

    def forward(self, x):
        """
        x: (N, in_dim) または (B, N, in_dim)
        DeepXDE からは (batch_size, in_dim) で来るので reshape が必要
        """
        if x.ndim == 2:
            # (N, in_dim) → (1, in_dim, N)
            x = x.unsqueeze(0).transpose(1, 2)
        elif x.ndim == 3:
            # (B, N, in_dim) → (B, in_dim, N)
            x = x.transpose(1, 2)

        # Pointwise conv
        f_local = self.feat(x)             # (B, C, N)
        f_global = torch.max(f_local, 2)[0]  # (B, C)

        # 各点にブロードキャストして局所特徴と結合
        f_global_expand = f_global.unsqueeze(-1).expand(-1, -1, f_local.shape[2])  # (B, C, N)
        f = torch.cat([f_local, f_global_expand], dim=1)  # (B, 2C, N)

        # (B, 2C, N) → (B*N, 2C)
        f = f.transpose(1, 2).reshape(-1, f.shape[1])
        y = self.head(f)  # (B*N, out_dim)
        return y




def main():
    # ---------- Load & scale ----------
    data = load_cfd_dataset()  # columns: [x,y,z,wd,u,v,w,p]
    features = ["x", "y", "z", "wd", "u", "v", "w", "p"]
    data_scaled, scalers = min_max_scale(data, features)

    x_in  = data_scaled[:, 0:1]
    y_in  = data_scaled[:, 1:2]
    z_in  = data_scaled[:, 2:3]
    wd_in = data_scaled[:, 3:4]
    u_in  = data_scaled[:, 4:5]
    v_in  = data_scaled[:, 5:6]
    w_in  = data_scaled[:, 6:7]
    p_in  = data_scaled[:, 7:8]

    # Inputs for PDE: (x,y,z,wd)
    ob_xyz = np.hstack((x_in, y_in, z_in, wd_in)).astype(np.float32)

    # ---------- Observations (PointSetBC) ----------
    rng = np.random.default_rng(42)
    num_anchor = min(5000, ob_xyz.shape[0])
    idx = rng.choice(ob_xyz.shape[0], num_anchor, replace=False)
    ob_xyz_sub = ob_xyz[idx]
    u_sub, v_sub, w_sub, p_sub = u_in[idx], v_in[idx], w_in[idx], p_in[idx]

    observe_u = dde.icbc.PointSetBC(ob_xyz_sub, u_sub, component=0)
    observe_v = dde.icbc.PointSetBC(ob_xyz_sub, v_sub, component=1)
    observe_w = dde.icbc.PointSetBC(ob_xyz_sub, w_sub, component=2)
    observe_p = dde.icbc.PointSetBC(ob_xyz_sub, p_sub, component=3)
    bcs = [observe_u, observe_v, observe_w, observe_p]

    # ---------- Geometry ----------
    geom = PointCloudGeometry(ob_xyz, tol=1e-5)

    # ---------- Network ----------
    
    net = PointNetLikeRegressor(
        in_dim=4,
        out_dim=4,
        local_widths=(128, 128, 128),
        head_widths=(256, 128),
        use_global_context=False,  # ← まずはFalse（混在バッチでも安全）
        initializer="he",
    )
    model = dde.Model(data, net)
    # ---------- DeepXDE data/model ----------
    data_pde = dde.data.PDE(
        geom,
        continuity_pde_noslip,
        bcs,
        num_domain=5000,
        anchors=ob_xyz_sub,
    )
    model = dde.Model(data_pde, net)

    # ---------- Compile & Train (NO callbacks) ----------
    print("Compiling model...")
    model.compile("adam", lr=1e-3, loss_weights=[1.0, 1.0] + [1.0] * len(bcs))
    print("\nTraining model (no callbacks)...")
    losshistory, train_state = model.train(
        iterations=50000,
        display_every=1000,
        disregard_previous_best=True,
    )

    # ---------- Save artifacts (PyTorch) ----------
    os.makedirs("artifacts", exist_ok=True)

    # Save PyTorch state_dict
    torch.save(model.net.state_dict(), "artifacts/model_state.pt")

    # Save minimal model config (for later reconstruction)
    config = {
        "layer_size": layer_size,
        "activation": activation,
        "initializer": initializer,
        "backend": "pytorch",
        "features": features,
    }
    with open("artifacts/model_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # Save DeepXDE training curves (csv/png)
    dde.saveplot(losshistory, train_state, issave=True, isplot=False, output_dir="artifacts")

    print("\n[INFO] Training complete. Saved:")
    print(" - artifacts/model_state.pt")
    print(" - artifacts/model_config.json")
    print(" - artifacts/loss_history*.csv (etc.)")

if __name__ == "__main__":
    main()
