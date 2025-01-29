from pointnet2_cls_ssg import get_model
from pointnet2_utils import PointNetSetAbstraction
import open3d as o3d
import numpy as np
import torch

pt_cloud = o3d.io.read_point_cloud("C:/Users/jmc_o/Documents/A5/INESC_TEC/dev/Datasets/L_scenario/A/pcl/pcl0.pcd")
xyz = np.asarray(pt_cloud.points)

sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=3, mlp=[64, 64, 128], group_all=False)
points, descriptors = 