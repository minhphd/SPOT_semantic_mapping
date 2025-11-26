import open3d as o3d
import argparse
import os

def view_mesh(mesh_path):
    if not os.path.exists(mesh_path):
        raise FileNotFoundError(mesh_path)
    
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()

    print(f"Loaded mesh: {mesh_path}")
    o3d.visualization.draw_geometries([mesh])

def view_pointcloud(pcd_path):
    if not os.path.exists(pcd_path):
        raise FileNotFoundError(pcd_path)

    pcd = o3d.io.read_point_cloud(pcd_path)

    print(f"Loaded point cloud: {pcd_path}")
    o3d.visualization.draw_geometries([pcd])

def view_both(mesh_path, pcd_path):
    if not os.path.exists(mesh_path) or not os.path.exists(pcd_path):
        raise FileNotFoundError("One or both paths do not exist.")

    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    pcd = o3d.io.read_point_cloud(pcd_path)

    print(f"Loaded mesh + point cloud")
    o3d.visualization.draw_geometries([mesh, pcd])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", "-m", type=str, default=None)
    parser.add_argument("--pcd", "-p", type=str, default=None)
    args = parser.parse_args()

    if args.mesh and args.pcd:
        view_both(args.mesh, args.pcd)
    elif args.mesh:
        view_mesh(args.mesh)
    elif args.pcd:
        view_pointcloud(args.pcd)
    else:
        print("Please provide --mesh <file> and/or --pcd <file>")
