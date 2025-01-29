from pointnet2_cls_ssg import get_model
import open3d as o3d
import numpy as np
import torch

pt_cloud = o3d.io.read_point_cloud("C:/Users/jmc_o/Documents/A5/INESC_TEC/dev/Datasets/L_scenario/A/pcl/pcl0.pcd")

model = get_model(num_class=40, normal_channel=False)
device = torch.device("cpu")
model.to(device)
model.eval()
model = torch.load("best_model.pth", map_location=device, weights_only=False)

points = np.array(pt_cloud.points)
points -= np.mean(points, axis=0)
points /= np.max(np.linalg.norm(points, axis=1))
points_tensor = torch.from_numpy(points).float().unsqueeze(0).transpose(2, 1)

# 4. Extract Features
with torch.no_grad():  # No need to calculate gradients
    model.eval()
    features, critical_indexes, A_feat = model(points_tensor)

