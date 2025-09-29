# main.py (PyTorch backend) — CFD PINN training with DeepXDE
import os, json
import numpy as np
import deepxde as dde

# --- Force PyTorch backend ---
dde.backend.set_default_backend("pytorch")
import torch

# --- Local utilities ---
from utils.user_define_function import (
    process_pipe_data,
    min_max_scale,
    inverse_scale,
    PointCloudGeometry,
    continuity_pde_noslip,
)

# ---------- Data loading ----------
DATA_PATH = "cfd_data.npy"  # assumes present in working dir
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"{DATA_PATH} not found. Place your CFD numpy file here.")

x1, y1, z1, wd1, u1, v1, w1, p1, flag1, data = process_pipe_data(DATA_PATH)

# ---------- Scaling ----------
features = ['x', 'y', 'z', 'wd', 'u', 'v', 'w', 'p']
data_scaled, scalers = min_max_scale(data, features)

x_in = data_scaled[:, 0:1]
y_in = data_scaled[:, 1:2]
z_in = data_scaled[:, 2:3]
wd_in = data_scaled[:, 3:4]
u_in = data_scaled[:, 4:5]
v_in = data_scaled[:, 5:6]
w_in = data_scaled[:, 6:7]
p_in = data_scaled[:, 7:8]

ob_xyz = np.hstack((x_in, y_in, z_in, wd_in)).astype(np.float32)

# ---------- Observations (PointSetBC) ----------
rng = np.random.default_rng(42)
num_anchor = min(5000, ob_xyz.shape[0])
idx = rng.choice(ob_xyz.shape[0], num_anchor, replace=False)
ob_xyz_sub = ob_xyz[idx]
u_in_sub, v_in_sub, w_in_sub, p_in_sub = u_in[idx], v_in[idx], w_in[idx], p_in[idx]

observe_u = dde.icbc.PointSetBC(ob_xyz_sub, u_in_sub, component=0)
observe_v = dde.icbc.PointSetBC(ob_xyz_sub, v_in_sub, component=1)
observe_w = dde.icbc.PointSetBC(ob_xyz_sub, w_in_sub, component=2)
observe_p = dde.icbc.PointSetBC(ob_xyz_sub, p_in_sub, component=3)

# Example pressure boundary at z ~= 0 (scaled space: feature index 2)
mask_z1 = np.isclose(ob_xyz[:, 2], 0.0, atol=1e-3)
ob_xyz_z1 = ob_xyz[mask_z1]
if ob_xyz_z1.size > 0:
    p_in_z1 = np.ones((ob_xyz_z1.shape[0], 1), dtype=np.float32)
    bc_z1 = dde.icbc.PointSetBC(ob_xyz_z1, p_in_z1, component=3)
    bcs = [observe_u, observe_v, observe_w, observe_p, bc_z1]
else:
    bcs = [observe_u, observe_v, observe_w, observe_p]

# ---------- Geometry ----------
geom = PointCloudGeometry(ob_xyz, tol=1e-5)

# ---------- Network ----------
layer_size = [4] + [128] * 8 + [4]
activation = "swish"
initializer = "He normal"
net = dde.maps.FNN(layer_size, activation, initializer)

# ---------- Problem & Model ----------
data_pde = dde.data.PDE(
    geom,
    continuity_pde_noslip,
    bcs,
    num_domain=5000,
    anchors=ob_xyz_sub,
)

model = dde.Model(data_pde, net)

# ---------- Training ----------
from deepxde.callbacks import EarlyStopping, ModelCheckpoint

early_stopping = EarlyStopping(
    monitor="loss",
    patience=5000,
    min_delta=1e-6,
    restore_best=True,
)

ckpt = ModelCheckpoint(
    filepath="artifacts/dde_model",
    save_better_only=True,
    period=1000,
    monitor="loss",
)

model.compile("adam", lr=1e-3, loss_weights=[1.0, 1.0] + [1.0] * len(bcs))
losshistory, train_state = model.train(
    iterations=50000,
    callbacks=[early_stopping, ckpt],
    display_every=1000,
    disregard_previous_best=True,
)

# ---------- Save artifacts (PyTorch) ----------
os.makedirs("artifacts", exist_ok=True)
# Save PyTorch weights
torch.save(model.net.state_dict(), "artifacts/model_state.pt")

# Save minimal architecture/config to reconstruct net
config = {
    "layer_size": layer_size,
    "activation": activation,
    "initializer": initializer,
    "backend": "pytorch",
    "features": features,
}
with open("artifacts/model_config.json", "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2)

# Optionally export TorchScript (best effort)
try:
    scripted = torch.jit.script(model.net)
    scripted.save("artifacts/model_scripted.pt")
except Exception as e:
    with open("artifacts/torchscript_error.txt","w",encoding="utf-8") as f:
        f.write(str(e))

# Save DeepXDE training curves
dde.saveplot(losshistory, train_state, issave=True, isplot=False, output_dir="artifacts")
print("Training complete. Artifacts saved in ./artifacts")