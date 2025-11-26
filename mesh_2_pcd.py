import open3d as o3d
import numpy as np
import argparse
import os


def load_existing_pcd(pcd_path):
    print(f"[Load] Loading existing point cloud: {pcd_path}")
    pcd = o3d.io.read_point_cloud(pcd_path)
    return pcd


def poisson_sample_mesh(mesh, num_points=500000, enable=True):
    if not enable:
        print("[Skip] Poisson sampling disabled.")
        return None
    
    print(f"[Poisson Sampling] Sampling {num_points} points from mesh...")
    pcd = mesh.sample_points_poisson_disk(
        number_of_points=num_points,
        init_factor=5
    )
    return pcd


def estimate_normals(pcd, radius=0.03, max_nn=30, enable=True):
    if not enable:
        print("[Skip] Normal estimation disabled.")
        return pcd
    
    print("[Normals] Estimating normals...")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius,
            max_nn=max_nn
        )
    )
    pcd.normalize_normals()
    return pcd


def reproject_to_mesh(pcd, mesh, enable=True):
    if not enable:
        print("[Skip] Reprojection disabled.")
        return pcd
    
    print("[Reprojection] Projecting points onto mesh surface...")

    mesh_tree = o3d.geometry.KDTreeFlann(mesh)
    pts = np.asarray(pcd.points)
    new_pts = np.zeros_like(pts)

    for i, p in enumerate(pts):
        _, idx, _ = mesh_tree.search_knn_vector_3d(p, 1)
        new_pts[i] = mesh.vertices[idx[0]]
        if i % 50000 == 0:
            print(f"  ... {i} points processed")

    pcd.points = o3d.utility.Vector3dVector(new_pts)
    return pcd


def smooth_pointcloud(pcd, iterations=1, k=30, enable=True):
    """
    Custom MLS-like smoothing that works on all Open3D versions.
    Uses k-NN plane projection. No deprecated APIs.
    """

    if not enable:
        print("[Skip] Smoothing disabled.")
        return pcd

    print(f"[Smoothing] Custom MLS smoothing ({iterations} iterations, k={k})...")

    points = np.asarray(pcd.points)
    tree = o3d.geometry.KDTreeFlann(pcd)

    for it in range(iterations):
        new_points = np.zeros_like(points)

        for i, p in enumerate(points):
            _, idx, _ = tree.search_knn_vector_3d(p, k)
            nbrs = points[idx]

            # Compute centroid
            centroid = np.mean(nbrs, axis=0)

            # PCA → normal
            cov = np.cov((nbrs - centroid).T)
            eigvals, eigvecs = np.linalg.eigh(cov)
            normal = eigvecs[:, 0]   # smallest eigenvector

            # Project point onto local plane
            proj = p - np.dot((p - centroid), normal) * normal

            new_points[i] = proj

        points = new_points
        pcd.points = o3d.utility.Vector3dVector(points)

        print(f"  Iteration {it+1}/{iterations} complete.")

        # rebuild KD-tree
        tree = o3d.geometry.KDTreeFlann(pcd)

    return pcd


def visualize(mesh, pcd):
    print("[View] Visualizing mesh + point cloud...")
    o3d.visualization.draw_geometries([mesh, pcd])


def main(args):
    assert os.path.exists(args.mesh), "Mesh file not found."

    print("[Load] Loading mesh...")
    mesh = o3d.io.read_triangle_mesh(args.mesh)
    mesh.compute_vertex_normals()

    # ======================================================
    # CASE 1: User provided an input PCD file → skip sampling
    # ======================================================
    if args.pcd is not None:
        assert os.path.exists(args.pcd), "Provided PCD file does not exist."
        print("[Mode] Using existing PCD file. Skipping Poisson sampling.")
        pcd = load_existing_pcd(args.pcd)

    # =======================================================
    # CASE 2: No PCD provided → do Poisson disk sampling
    # =======================================================
    else:
        pcd = poisson_sample_mesh(mesh, args.num_points, enable=args.sample)
        if pcd is None:
            raise ValueError("No PCD provided and Poisson sampling disabled. Nothing to process.")

    # 2. Normal estimation
    pcd = estimate_normals(
        pcd,
        radius=args.normal_radius,
        max_nn=args.normal_nn,
        enable=args.normals
    )

    # 3. Re-project to mesh
    pcd = reproject_to_mesh(pcd, mesh, enable=args.reproject)

    # 4. Smoothing
    pcd = smooth_pointcloud(
        pcd,
        iterations=args.smooth_iters,
        enable=args.smooth
    )

    # Save results
    if args.output:
        print(f"[Save] Saving final point cloud to {args.output}")
        o3d.io.write_point_cloud(args.output, pcd)

    # Visualize
    visualize(mesh, pcd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--mesh", type=str, required=True, help="Input mesh path")
    parser.add_argument("--pcd", type=str, default=None, help="Optional input PCD file. If provided, Poisson sampling is skipped.")
    parser.add_argument("--output", type=str, default="output_pcd.ply", help="Output point cloud path")

    # 1. Poisson sampling
    parser.add_argument("--sample", action="store_true", help="Enable Poisson sampling (ignored if --pcd is provided)")
    parser.add_argument("--num_points", type=int, default=500000)

    # 2. Normal estimation
    parser.add_argument("--normals", action="store_true")
    parser.add_argument("--normal_radius", type=float, default=0.03)
    parser.add_argument("--normal_nn", type=int, default=30)

    # 3. Reprojection
    parser.add_argument("--reproject", action="store_true")

    # 4. Smoothing
    parser.add_argument("--smooth", action="store_true")
    parser.add_argument("--smooth_iters", type=int, default=1)

    args = parser.parse_args()
    main(args)
