import open3d as o3d
import numpy as np
from randlanet_model import RandLANet  # your model import

pcd = o3d.io.read_point_cloud("clouds\\hybrid_pointcloud.ply")
pts = np.asarray(pcd.points)

# downsample for speed
pcd_down = pcd.voxel_down_sample(0.03)
pts_down = np.asarray(pcd_down.points)

pred_labels = model.predict(pts_down)   # Nx1 segmentation labels

# colorize based on label
colors = label_to_color(pred_labels)
pcd_down.colors = o3d.utility.Vector3dVector(colors)

o3d.io.write_point_cloud("semseg.ply", pcd_down)