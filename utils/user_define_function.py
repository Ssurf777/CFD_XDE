# utils/user_define_function.py
# Consolidated utilities for DeepXDE + PyTorch backend PINN (CFD)
import numpy as np
import pandas as pd
import open3d as o3d
import deepxde as dde
from scipy.spatial import cKDTree

# ---- Ensure PyTorch backend ----
dde.backend.set_default_backend("pytorch")
import torch

# ---------- Scaling utilities ----------
def min_max_scale(data: np.ndarray, feature_names):
    data_scaled = data.copy().astype(np.float32)
    scalers = {}
    for i, name in enumerate(feature_names):
        col = data[:, i].astype(np.float64)
        min_val = float(np.min(col))
        max_val = float(np.max(col))
        scale = float(max_val - min_val)
        if np.isclose(scale, 0.0):
            scale = 1.0
        data_scaled[:, i] = (col - min_val) / scale
        scalers[name] = {"min": min_val, "scale": scale}
    return data_scaled, scalers

def inverse_scale(value_scaled, name, scalers):
    s = scalers[name]
    return value_scaled * s["scale"] + s["min"]

# ---------- Geometry from point cloud (4D: x,y,z,wd) ----------
class PointCloudGeometry(dde.geometry.Geometry):
    def __init__(self, points: np.ndarray, tol: float = 1e-4):
        """
        points: (N,4) expected -> x,y,z,wd
        tol:    nearest-neighbor radius tolerance for 'inside' predicate
        """
        assert points.ndim == 2 and points.shape[1] == 4, "PointCloudGeometry expects (N,4): x,y,z,wd"
        self.points = points.astype(np.float32)
        self.tol = tol
        # KDTree over all 4 dims (x,y,z,wd). wd helps distinguish interior vs wall vicinity anchor spacing
        self.kdtree = cKDTree(self.points)

        bbox = np.array([np.min(self.points, axis=0), np.max(self.points, axis=0)], dtype=np.float32)
        diam = float(np.linalg.norm(bbox[1] - bbox[0]))
        super().__init__(dim=4, bbox=bbox, diam=diam)

    def inside(self, x):
        # inside if close to some anchor point
        dist, _ = self.kdtree.query(x, k=1)
        return dist < self.tol

    def on_boundary(self, x):
        dist, _ = self.kdtree.query(x, k=1)
        return np.logical_and(dist >= self.tol * 0.9, dist <= self.tol * 1.1)

    def random_points(self, n, random="pseudo"):
        idx = np.random.choice(len(self.points), n, replace=False)
        return self.points[idx]

    def random_boundary_points(self, n, random="pseudo"):
        raise NotImplementedError("Boundary sampling not implemented for point cloud geometry.")

# ---------- PDE residuals: continuity + soft no-slip via wd ----------
WALL_WIDTH = 0.01  # fraction (0-1) in scaled wd for wall band

def continuity_pde_noslip(x, y):
    """
    x: (N,4) = [x,y,z,wd] in scaled domain
    y: (N,4) = [u,v,w,p]
    returns: [continuity_residual, no_slip_soft_residual]
    """
    # Split fields
    u, v, w = y[:, 0:1], y[:, 1:2], y[:, 2:3]

    # Continuity (∂u/∂x + ∂v/∂y + ∂w/∂z). Coordinates assumed already scaled.
    du_dx = dde.grad.jacobian(y, x, i=0, j=0)
    dv_dy = dde.grad.jacobian(y, x, i=1, j=1)
    dw_dz = dde.grad.jacobian(y, x, i=2, j=2)
    continuity = du_dx + dv_dy + dw_dz  # (N,1)

    # Soft no-slip mask near wall using wd (x[:,3])
    wd = x[:, 3:4]
    mask_wall = torch.clamp((WALL_WIDTH - wd) / WALL_WIDTH, min=0.0, max=1.0)

    speed_norm = torch.linalg.norm(torch.cat([u, v, w], dim=1), dim=1, keepdim=True)
    wall_res = mask_wall * speed_norm  # (N,1)

    return [continuity, wall_res]

# ---------- CFD data processing ----------
def process_pipe_data(file_path, threshold_mesh: float = 0.001):
    """
    Args:
        file_path: npy file path. Columns: x,y,z,u,v,w,p (Nx7)
        threshold_mesh: (meters in original scale) distance gate for wall vertex to NN point mapping
    Returns:
        x_in,y_in,z_in,wd_in,u_in,v_in,w_in,p_in,flag_in,data (like original user function)
    """
    data = np.load(file_path)
    x, y, z, u, v, w, p = [data[:, i] for i in range(7)]
    points = np.column_stack((x, y, z))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Base wall mask from near-zero velocities
    tol = 0.01
    wall_mask = (np.abs(u) < tol) & (np.abs(v) < tol) & (np.abs(w) < tol)
    wall_points = np.column_stack((x[wall_mask], y[wall_mask], z[wall_mask]))
    wall_pcd = o3d.geometry.PointCloud()
    wall_pcd.points = o3d.utility.Vector3dVector(wall_points)

    distances = pcd.compute_point_cloud_distance(wall_pcd)
    wd = np.asarray(distances)

    df = pd.DataFrame({'x': x, 'y': y, 'z': z, 'wd': wd, 'u': u, 'v': v, 'w': w, 'p': p})
    data_np = df.to_numpy()

    # Alpha-shape to refine wall indices
    xyz = data_np[:, :3]
    pcd_full = o3d.geometry.PointCloud()
    pcd_full.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    dists = np.asarray(pcd_full.compute_nearest_neighbor_distance())
    d_mean = float(np.mean(dists))
    alpha = 2.5 * d_mean
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd_full, alpha)
    mesh.compute_vertex_normals()
    mesh_vertices = np.asarray(mesh.vertices)

    # KDTree in xyz space
    from sklearn.neighbors import KDTree as SkKDTree
    kdt = SkKDTree(xyz)
    wall_indices = []
    for v in mesh_vertices:
        dist, ind = kdt.query([v], k=1, return_distance=True)
        if dist[0,0] < threshold_mesh:
            wall_indices.append(ind[0,0])
    wall_indices = np.unique(wall_indices)
    wall_xyz = xyz[wall_indices]

    # Flag points that are in wall set
    def rows_in_a_in_b(a, b):
        # Works for float rows
        a_view = np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
        b_view = np.ascontiguousarray(b).view(np.dtype((np.void, b.dtype.itemsize * b.shape[1])))
        return np.in1d(a_view, b_view)

    flag_in = rows_in_a_in_b(xyz, wall_xyz).astype(int).reshape(-1, 1)

    # Split
    x_in  = data_np[:, 0].reshape(-1, 1)
    y_in  = data_np[:, 1].reshape(-1, 1)
    z_in  = data_np[:, 2].reshape(-1, 1)
    wd_in = data_np[:, 3].reshape(-1, 1)
    u_in  = data_np[:, 4].reshape(-1, 1)
    v_in  = data_np[:, 5].reshape(-1, 1)
    w_in  = data_np[:, 6].reshape(-1, 1)
    p_in  = data_np[:, 7].reshape(-1, 1)

    return x_in, y_in, z_in, wd_in, u_in, v_in, w_in, p_in, flag_in, data_np