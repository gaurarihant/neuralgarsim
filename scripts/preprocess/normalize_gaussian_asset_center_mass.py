import argparse
import os
import glob
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

    center = mesh.center_mass
    scale = 1.0 / mesh.scale

    mesh.vertices -= center
    mesh.vertices *= scale

    T = np.vstack([np.hstack([R_mat, np.zeros((3, 1))]), [0, 0, 0, 1]])
    mesh.apply_transform(T)
    
    mesh.export(out_obj_path)
    print(f"Saved Mesh: {out_obj_path}")

    if ply_path and os.path.exists(ply_path):
        process_gaussians(ply_path, out_ply_path, center, scale, R_mat)


def main():
    parser = argparse.ArgumentParser(description="Normalise paired mesh and 3D Gaussian PLY assets using mesh centre of mass.")
    parser.add_argument("--input-dir", required=True, help="directory containing OBJ files")
    parser.add_argument("--output-dir", default="data/garments/processed")
    parser.add_argument("--prefix", default="gaussians_")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    obj_files = sorted(glob.glob(os.path.join(args.input_dir, "*.obj")))
    print(f"Found {len(obj_files)} OBJ files")

    for obj in obj_files:
        base_obj = os.path.basename(obj)
        out_obj = os.path.join(args.output_dir, f"{args.prefix}{base_obj}")

        ply_path = obj.replace("garment_mesh.obj", "gaussians.ply") if "garment_mesh" in base_obj else obj.replace(".obj", ".ply")
        out_ply = os.path.join(args.output_dir, f"{args.prefix}{os.path.basename(ply_path)}")
        if not os.path.exists(ply_path):
            ply_path = None

        process_pair(obj, ply_path, out_obj, out_ply)

    print("\nAll processing complete.\n")


if __name__ == "__main__":
    main()
