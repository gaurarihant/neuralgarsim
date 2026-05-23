import argparse
import numpy as np
import trimesh
from scipy.spatial import cKDTree

def referenced_vertices(mesh):
    used_idx = np.unique(mesh.faces.flatten())
    return used_idx, mesh.vertices[used_idx]

def map_boundary(boundary_positions, mesh):
    used_idx, vertices = referenced_vertices(mesh)
    tree = cKDTree(vertices)
    dists, idxs = tree.query(boundary_positions)
    mapped_indices = used_idx[idxs]
    return mapped_indices, dists

def main():
    parser = argparse.ArgumentParser(description="Map boundary vertices from one mesh to remeshed targets.")
    parser.add_argument("--source", required=True, help="source OBJ path")
    parser.add_argument("--targets", required=True, nargs="+", help="target OBJ paths")
    parser.add_argument("--boundary-indices", required=True, type=int, nargs="+")
    args = parser.parse_args()

    source = trimesh.load(args.source, process=False)
    boundary_positions = source.vertices[args.boundary_indices]

    print("Source vertices:", len(source.vertices))
    print("Source faces:", len(source.faces))
    print("Boundary positions:\n", boundary_positions)
    print("\nMapping results")

    for target_path in args.targets:
        target = trimesh.load(target_path, process=False)
        mapped, dists = map_boundary(boundary_positions, target)
        print(target_path)
        print("Mapped indices:", mapped.tolist())
        print(f"Mean distance: {np.mean(dists):.6f}")


if __name__ == "__main__":
    main()
