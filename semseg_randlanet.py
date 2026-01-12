import os
import glob
import numpy as np
import open3d as o3d
import open3d.ml as _ml3d
import open3d.ml.torch as ml3d

# ------------------------------------------
# VISUALIZATION UTILITY
# ------------------------------------------
def custom_draw_geometry(pcd):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    opt = vis.get_render_option()
    opt.point_size = 2.0
    opt.background_color = np.asarray([1.0, 1.0, 1.0])
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()

# ------------------------------------------
# LOAD PCD FILES FROM A FOLDER
# ------------------------------------------
def load_custom_dataset(folder):
    paths = sorted(glob.glob(folder + "/*.pcd"))
    print(f"Loaded {len(paths)} point clouds.")
    pcds = [o3d.io.read_point_cloud(p) for p in paths]
    return pcds

# ------------------------------------------
# PREPARE FOR PIPELINE
# ------------------------------------------
def align_to_z_up(xyz):
    # SPOT's coordinate may have Y-up or X-up depending on source.
    # Adjust these based on your dataset.
    # Here's a common transform: Y-up → Z-up

    R = np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, -1, 0]
    ])  # rotates Y-up → Z-up

    return xyz @ R.T

def fix_num_points(xyz, rgb, num_points=40960):
    N = xyz.shape[0]

    if N >= num_points:
        idx = np.random.choice(N, num_points, replace=False)
        return xyz[idx], rgb[idx]
    else:
        pad = num_points - N
        xyz_pad = np.pad(xyz, ((0, pad), (0, 0)), mode='edge')
        rgb_pad = np.pad(rgb, ((0, pad), (0, 0)), mode='edge')
        return xyz_pad, rgb_pad

def normalize_height(xyz):
    xyz[:, 2] = xyz[:, 2] - np.min(xyz[:, 2])  # floor at 0
    return xyz

def normalize_rgb(colors):
    colors = np.asarray(colors, dtype=np.float32)
    if colors.max() > 1.0:
        colors = colors / 255.0
    return (colors - 0.5) * 2.0   # same as (x - 128)/128 but normalized 0–255

def voxel_downsample(xyz, rgb, voxel_size=0.06):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    pcd = pcd.voxel_down_sample(voxel_size)
    xyz = np.asarray(pcd.points)
    rgb = np.asarray(pcd.colors)
    return xyz, rgb

def shuffle_points(xyz, rgb):
    idx = np.random.permutation(xyz.shape[0])
    return xyz[idx], rgb[idx]

def preprocess_like_s3dis(xyz, colors, num_points=40960):
    # 1) Align
    xyz = align_to_z_up(xyz)

    # 2) Normalize height
    xyz = normalize_height(xyz)

    # 3) Normalize RGB
    colors = normalize_rgb(colors)

    # 4) Voxel downsample
    xyz, colors = voxel_downsample(xyz, colors, voxel_size=0.06)

    # 5) Fix number of points
    xyz, colors = fix_num_points(xyz, colors, num_points)

    # 6) Shuffle
    xyz, colors = shuffle_points(xyz, colors)

    return xyz, colors

def prepare_point_cloud_for_inference(pcd, num_points=40960):
    pcd.remove_non_finite_points()

    xyz = np.asarray(pcd.points).astype(np.float32)
    colors = np.asarray(pcd.colors)

    if colors.shape[0] == 0:
        colors = np.zeros_like(xyz)

    # 🔥 Apply full S3DIS preprocessing
    xyz, colors = preprocess_like_s3dis(xyz, colors, num_points)

    data = {
        "name": "custom",
        "point": xyz,
        "feat": colors.astype(np.float32),
        "label": np.zeros((len(xyz),), dtype=np.int32),
    }
    return data, pcd

# --------------------------------------------------------
# COLOR MAP (13 S3DIS classes used by pretrained RandLANet)
# --------------------------------------------------------
COLOR_MAP = {
    0: (0,0,0),
    1: (245,150,100),
    2: (245,230,100),
    3: (150,60,30),
    4: (180,30,80),
    5: (255,0,0),
    6: (30,30,255),
    7: (200,40,255),
    8: (90,30,150),
    9: (255,0,255),
    10: (255,150,255),
    11: (75,0,75),
    12: (75,0,175),
}
for key in COLOR_MAP:
    COLOR_MAP[key] = tuple(c/255 for c in COLOR_MAP[key])

#
# ==============================
#        MAIN SCRIPT
# ==============================
#

# Load config
cfg_file = "configs/randlanet.yml"
cfg = _ml3d.utils.Config.load_from_file(cfg_file)

# Load model
model = ml3d.models.RandLANet(**cfg.model)

# Create pipeline WITHOUT DATASET
pipeline = ml3d.pipelines.SemanticSegmentation(
    model=model,
    dataset=None,
    device="cpu",
    **cfg.pipeline
)

# Download checkpoint if needed
ckpt_folder = "./logs/"
os.makedirs(ckpt_folder, exist_ok=True)
ckpt_path = ckpt_folder + "randlanet_s3dis_202201071330utc.pth"
url = "https://storage.googleapis.com/open3d-releases/model-zoo/randlanet_s3dis_202201071330utc.pth"

if not os.path.exists(ckpt_path):
    print("Downloading pretrained RandLANet weights...")
    os.system(f"wget {url} -O {ckpt_path}")

pipeline.load_ckpt(ckpt_path)

# -------------------------------------
# LOAD YOUR PCD FOLDER
# -------------------------------------
pcd_folder = "./clouds"
clouds = load_custom_dataset(pcd_folder)

# -------------------------------------
# RUN INFERENCE ON ANY CLOUD
# -------------------------------------
pcd_index = 1
data, pcd = prepare_point_cloud_for_inference(clouds[pcd_index])
o3d.io.write_point_cloud("segmented_output.pcd", pcd)
raise
print("Running inference...")
result = pipeline.run_inference(data)

pred = result["predict_labels"]

# Colorize
pcd.colors = o3d.utility.Vector3dVector([COLOR_MAP[int(c)] for c in pred])

# # Visualize
# custom_draw_geometry(pcd)
o3d.io.write_point_cloud("segmented_output.pcd", pcd)

print("Done.")
