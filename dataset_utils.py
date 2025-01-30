import numpy as np
import open3d as o3d
import torch
import os


def load_pose(pose_file):
    return np.loadtxt(pose_file)

def extract_features(pcd_path, PointNet2):
    # Load Point Cloud
    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points)

    # Check for NaN and Inf in the original points
    has_nan = np.isnan(points).any()
    has_inf = np.isinf(points).any()

    if has_nan or has_inf:
      # Handle NaN and Inf values
      # Option 1: Remove points with NaN or Inf
      valid_mask = ~np.isnan(points).any(axis=1) & ~np.isinf(points).any(axis=1)
      points = points[valid_mask]

    points -= np.mean(points, axis=0)
    norm = np.linalg.norm(points, axis=1)
    points /= (np.max(norm) + 1e-8)  # Avoid division by zero

    points_tensor = torch.from_numpy(points).float().unsqueeze(0).transpose(2, 1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    PointNet2 = PointNet2.to(device)
    points_tensor = points_tensor.to(device)

    with torch.no_grad():
        features = PointNet2(points_tensor, True)
    return features