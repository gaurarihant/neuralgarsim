import argparse
import os
import glob

import numpy as np
import trimesh


def mesh_centre(vertices, method, np):
    if method == "mean":
        return vertices.mean(axis=0)
    if method == "bounds":
        return 0.5 * (vertices.min(axis=0) + vertices.max(axis=0))
    if method == "centre_mass":
        return None
    raise ValueError(f"Unknown centre method: {method}")


def axis_transform_matrix(name):
    if name == "none":
        return np.eye(3)
    if name == "y_to_z":
        return np.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0],
        ])
    if name == "y_to_minus_z":
        return np.array([
            [1, 0, 0],
            [0, 0, 1],
            [0, -1, 0],
        ])
    raise ValueError(f"Unknown axis transform: {name}")


def process_mesh(
    obj_path,
    out_path,
    centre_method="mean",
    target_max_abs=1.0,
    axis_transform="none",
):

    if not os.path.exists(obj_path):
        raise FileNotFoundError(f"Input mesh not found: {obj_path}")

    with open(obj_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    vertex_lines = []
    vertices = []
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if not stripped.startswith("v "):
            continue
        parts = stripped.split()
        if len(parts) < 4:
            raise ValueError(f"Malformed vertex line {i + 1} in {obj_path}")
        vertex_lines.append(i)
        vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])

    if not vertices:
        raise ValueError(f"No OBJ vertex records found in {obj_path}")

    vertices = np.asarray(vertices, dtype=np.float64)

    if centre_method == "centre_mass":
        mesh = trimesh.load(obj_path, force="mesh", process=False, maintain_order=True)
        centre = mesh.center_mass
    else:
        centre = mesh_centre(vertices, centre_method, np)

    vertices = vertices - centre

    max_abs = np.abs(vertices).max()
    if max_abs <= 0:
        raise ValueError(f"Cannot normalise degenerate mesh: {obj_path}")
    vertices *= target_max_abs / max_abs

    R = axis_transform_matrix(axis_transform)
    vertices = vertices @ R.T

    for line_index, vertex in zip(vertex_lines, vertices):
        original = lines[line_index]
        newline = "\n" if original.endswith("\n") else ""
        lines[line_index] = f"v {vertex[0]:.10g} {vertex[1]:.10g} {vertex[2]:.10g}{newline}"

    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"Saved: {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Normalise garment OBJ assets.")
    parser.add_argument("--input", help="single OBJ file to process")
    parser.add_argument("--input-dir", help="directory containing OBJ files to process")
    parser.add_argument("--output-dir", default="data/garments/processed")
    parser.add_argument("--prefix", default="", help="optional output filename prefix")
    parser.add_argument("--centre-method", choices=["mean", "bounds", "centre_mass"], default="mean")
    parser.add_argument("--target-max-abs", type=float, default=1.0)
    parser.add_argument("--axis-transform", choices=["none", "y_to_z", "y_to_minus_z"], default="none")
    args = parser.parse_args()

    if not args.input and not args.input_dir:
        parser.error("Provide --input or --input-dir.")

    os.makedirs(args.output_dir, exist_ok=True)
    obj_files = [args.input] if args.input else sorted(glob.glob(os.path.join(args.input_dir, "*.obj")))

    for obj in obj_files:
        base = os.path.basename(obj)
        out_path = os.path.join(args.output_dir, f"{args.prefix}{base}")
        process_mesh(
            obj,
            out_path,
            centre_method=args.centre_method,
            target_max_abs=args.target_max_abs,
            axis_transform=args.axis_transform,
        )

    print("\nAll processing complete.\n")


if __name__ == "__main__":
    main()
