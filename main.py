from pointnet2_cls_ssg import get_model
from pointnet2_utils import PointNetSetAbstraction
import open3d as o3d
import numpy as np
import torch
import os
from dataset_utils import load_pose, extract_features
import time

root_dirs = [
    'C:/Users/jmc_o/Documents/A5/INESC_TEC/dev/Datasets/L_scenario/A',
   'C:/Users/jmc_o/Documents/A5/INESC_TEC/dev/Datasets/L_scenario/B'
]

# Load the checkpoint
checkpoint = torch.load('best_model.pth', map_location=torch.device('cpu'))
# Extract the model state dictionary
model_state_dict = checkpoint['model_state_dict']

model = get_model(num_class=40, normal_channel=False)  # The num_class here is a placeholder
model.load_state_dict(model_state_dict)
model.eval()


# KD-tree construction
descriptors = []
poses = []

# Iterate through root directories
i = 0
for root_dir in root_dirs:
    pcl_dir = os.path.join(root_dir, 'pcl')
    pose_dir = os.path.join(root_dir, 'pose')

    pcl_files = sorted(os.listdir(pcl_dir))
    pose_files = sorted(os.listdir(pose_dir))

    for pcl_file, pose_file in zip(pcl_files, pose_files):
        start_time = time.time()
        pcl_path = os.path.join(pcl_dir, pcl_file)
        pose_path = os.path.join(pose_dir, pose_file)
        pose = load_pose(pose_path)
        features = extract_features(pcl_path, model)
        descriptors.append(features)
        poses.append(pose)
        end_time = time.time()
        print(f"Processed {pcl_file} in {end_time - start_time} seconds.")

# Move tensors to CPU and convert to NumPy arrays
descriptors = [desc.cpu().numpy() for desc in descriptors]
poses = [pose.cpu().numpy() if torch.is_tensor(pose) else pose for pose in poses]  # Handle tensors or non-tensors

squeezed_descriptors = [desc.squeeze(0) for desc in descriptors]

np.save('poses.npy', poses)
print("poses saved successfully.")
np.save('descriptors_1024_sa3_cls_ssg.npy', squeezed_descriptors)
print("descriptors saved successfully.")