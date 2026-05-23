import os
import imageio
import torch
import numpy as np
from pytorch3d.io import save_obj
from .config import device 

def save_ply_with_color(filename, verts, faces, verts_rgb):
    verts = verts.detach().cpu().numpy()
    faces = faces.detach().cpu().numpy()
    verts_rgb = verts_rgb.detach().cpu().numpy()
    
    n_verts = verts.shape[0]
    n_faces = faces.shape[0]
    n_colors = verts_rgb.shape[0]
    
    if n_verts != n_colors:
        print(f"Warning: Verts {n_verts} and Colors {n_colors} mismatch. Truncating to min.")
        n_verts = min(n_verts, n_colors)
        verts = verts[:n_verts]
        verts_rgb = verts_rgb[:n_verts]

    verts_rgb = (verts_rgb * 255).clip(0, 255).astype(np.uint8)
    
    # Meshlab format
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n_verts}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write(f"element face {n_faces}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")

        for i in range(n_verts):
            v = verts[i]
            c = verts_rgb[i]
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {c[0]} {c[1]} {c[2]}\n")
            
        # Write Faces
        for i in range(n_faces):
            fa = faces[i]
            f.write(f"3 {int(fa[0])} {int(fa[1])} {int(fa[2])}\n")

def save_meshes(positions, faces, meshes_dir, i, verts_uvs=None, tex_image_file=None, other_meshes=[], verts_rgb=None, suffix=""):
    
    os.makedirs(meshes_dir, exist_ok=True)

    for idx, mesh in enumerate(other_meshes):
        body_path = os.path.join(meshes_dir, f"0/body_{idx}.obj")
        if not os.path.exists(body_path) or i==0:
            os.makedirs(os.path.join(meshes_dir, "0"), exist_ok=True)
            save_obj(body_path, mesh.verts_packed(), mesh.faces_packed())

    meshes_dir = os.path.join(meshes_dir, f'{i}')
    os.makedirs(meshes_dir, exist_ok=True)
    
    if verts_rgb is not None:
        filename = f'{0:03d}{suffix}.ply'
        save_ply_with_color(
            os.path.join(meshes_dir, filename), 
            positions[0],
            faces, 
            verts_rgb
        )
    else:
        if tex_image_file is not None:
            try:
                texture_map = torch.tensor(imageio.imread(tex_image_file)/255., dtype=torch.float, device=device)
            except:
                texture_map = None
        else:
            texture_map = None
        
        filename = f'{0:03d}{suffix}.obj'
        save_obj(
            os.path.join(meshes_dir, filename), 
            positions[0], 
            faces, 
            verts_uvs=verts_uvs, 
            faces_uvs=faces, 
            texture_map=texture_map
        )