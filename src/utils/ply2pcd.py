import open3d as o3d

def ply_to_pcd(input_path="input.ply", output_path="output.pcd"):
    # Load your .ply
    pcd = o3d.io.read_point_cloud(input_path)

    # Save as .pcd (binary)
    o3d.io.write_point_cloud(output_path, pcd, write_ascii=False)

if __name__ == "__main__":
    # take in user input and output paths
    import argparse
    parser = argparse.ArgumentParser(description="Convert PLY to PCD format.")
    parser.add_argument("input", type=str, help="Input PLY file path")
    parser.add_argument("output", type=str, help="Output PCD file path")
    args = parser.parse_args()
    ply_to_pcd(args.input, args.output)