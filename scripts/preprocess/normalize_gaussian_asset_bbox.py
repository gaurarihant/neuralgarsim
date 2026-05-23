import argparse
import os
import trimesh
import numpy as np
from plyfile import PlyData
from scipy.spatial.transform import Rotation

R_mat = np.array([
    [1,  0,  0],
    [0,  0, -1],
    [0,  1,  0]
])

def process_gaussians(ply_path, out_path, center, scale, R):
    """Applies spatial transformations to 3DGS PLY data."""
    plydata = PlyData.read(ply_path)
    v = plydata['vertex']
    
    pos = np.vstack([v['x'], v['y'], v['z']]).T
    pos = (pos - center) * scale
    pos = pos @ R.T
    v['x'], v['y'], v['z'] = pos[:, 0], pos[:, 1], pos[:, 2]
    
    normals = np.vstack([v['nx'], v['ny'], v['nz']]).T
    normals = normals @ R.T
    v['nx'], v['ny'], v['nz'] = normals[:, 0], normals[:, 1], normals[:, 2]
    
    log_scale = np.log(scale)
    v['scale_0'] += log_scale
    v['scale_1'] += log_scale
    v['scale_2'] += log_scale
    
    quats_old = np.vstack([v['rot_1'], v['rot_2'], v['rot_3'], v['rot_0']]).T 
    
    rot_spatial = Rotation.from_matrix(R)
    rot_old = Rotation.from_quat(quats_old)
    
    rot_new = rot_spatial * rot_old
    quats_new = rot_new.as_quat()
    
    v['rot_1'] = quats_new[:, 0]
    v['rot_2'] = quats_new[:, 1]
    v['rot_3'] = quats_new[:, 2]
    v['rot_0'] = quats_new[:, 3]
    
    PlyData([plydata.elements[0]]).write(out_path)
    print(f"Saved Gaussians: {out_path}")


def process_pair(obj_path, ply_path, out_obj_path, out_ply_path):
    mesh = trimesh.load(obj_path, force='mesh', process=False)

    min_bound, max_bound = mesh.bounds
    center = (min_bound + max_bound) / 2.0

    extents = max_bound - min_bound
    max_extent = np.max(extents)
    scale = 1.0 / max_extent

    print(f"Processing {os.path.basename(obj_path)}:")
    print(f"  - Original centre: {center}")
    print(f"  - Applied Scale:   {scale:.4f}")

    mesh.vertices -= center
    mesh.vertices *= scale

    T = np.eye(4)
    T[:3, :3] = R_mat
    mesh.apply_transform(T)
    
    mesh.export(out_obj_path)
    print(f"  - Saved Mesh: {out_obj_path}")

    if ply_path and os.path.exists(ply_path):
        process_gaussians(ply_path, out_ply_path, center, scale, R_mat)


def main():
    parser = argparse.ArgumentParser(description="Normalise a paired mesh and 3D Gaussian PLY asset using the mesh bounding box.")
    parser.add_argument("--obj", required=True, help="input OBJ path")
    parser.add_argument("--ply", help="input Gaussian PLY path")
    parser.add_argument("--out-obj", default="data/garments/template_processed.obj")
    parser.add_argument("--out-ply", default="data/garments/point_cloud_processed.ply")
    args = parser.parse_args()

    process_pair(args.obj, args.ply, args.out_obj, args.out_ply)
    print("\nAll processing complete.\n")


if __name__ == "__main__":
    main()
