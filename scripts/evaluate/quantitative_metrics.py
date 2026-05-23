import argparse
import json
import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree
import trimesh


def _cotangent_laplacian(verts, faces):
    N = len(verts)
    row, col, data = [], [], []
    for k in range(3):
        i = faces[:, (k + 1) % 3]
        j = faces[:, (k + 2) % 3]
        opp = faces[:, k]
        vi = verts[i] - verts[opp]
        vj = verts[j] - verts[opp]
        dot = np.einsum('ij,ij->i', vi, vj)
        cross_norm = np.maximum(
            np.linalg.norm(np.cross(vi, vj), axis=1), 1e-12)
        w = 0.5 * dot / cross_norm
        row.extend(i); col.extend(j); data.extend(-w)
        row.extend(j); col.extend(i); data.extend(-w)
    L = sp.csr_matrix((data, (row, col)), shape=(N, N))
    L += sp.diags(np.array(-L.sum(axis=1)).flatten())
    return L


def _vertex_areas(verts, faces):
    N = len(verts)
    fa = 0.5 * np.linalg.norm(
        np.cross(verts[faces[:, 1]] - verts[faces[:, 0]],
                 verts[faces[:, 2]] - verts[faces[:, 0]]), axis=1)
    A = np.zeros(N)
    for k in range(3):
        np.add.at(A, faces[:, k], fa / 3.0)
    return A


def mean_curvature(mesh: trimesh.Trimesh):
    """Returns (H, A) — absolute mean curvature and vertex areas."""
    V, F = np.array(mesh.vertices), np.array(mesh.faces)
    L = _cotangent_laplacian(V, F)
    A = _vertex_areas(V, F)
    H = np.linalg.norm(L @ V, axis=1) / (2.0 * np.maximum(A, 1e-12))
    return np.abs(H), A


def _interior_mask(verts, margin=0.05):
    bmin, bmax = verts.min(0), verts.max(0)
    ext = margin * (bmax - bmin)
    return np.all((verts > bmin + ext) & (verts < bmax - ext), axis=1)


def _chamfer(Va, Vb):
    d_ab, _ = cKDTree(Vb).query(Va)
    d_ba, _ = cKDTree(Va).query(Vb)
    return float(np.mean(d_ab) + np.mean(d_ba))


def wrinkle_chamfer(mesh_a: trimesh.Trimesh,
                    mesh_b: trimesh.Trimesh,
                    percentile: float = 75.0) -> dict:
    """
    Chamfer distance computed only on high-curvature (wrinkly) vertices.

    If wrinkle LOCATIONS shift between discretisations, this will be large
    even if total curvature magnitude stays similar. This is what captures
    the visual 'wrinkle shift' that the plain CV metric misses.

    Each mesh uses its own p{percentile} threshold so scale differences
    between methods don't affect the comparison.
    """
    Va = np.array(mesh_a.vertices)
    Vb = np.array(mesh_b.vertices)
    Ha, _ = mean_curvature(mesh_a)
    Hb, _ = mean_curvature(mesh_b)

    wrinkle_a = Va[Ha >= np.percentile(Ha, percentile)]
    wrinkle_b = Vb[Hb >= np.percentile(Hb, percentile)]

    return {
        'wrinkle_chamfer':    _chamfer(wrinkle_a, wrinkle_b),
        'full_chamfer':       _chamfer(Va, Vb),
        'n_wrinkle_verts_a':  len(wrinkle_a),
        'n_wrinkle_verts_b':  len(wrinkle_b),
    }


def curvature_correlation(mesh_a: trimesh.Trimesh,
                          mesh_b: trimesh.Trimesh) -> float:
    """
    Pearson correlation of curvature fields after nearest-neighbour transfer.

    Transfer mesh_b's curvature onto mesh_a's vertex positions, then
    correlate with mesh_a's curvature. High value = wrinkle patterns
    appear in the same spatial locations, regardless of mesh resolution.
    Range: [-1, 1]. Higher is better (more consistent pattern).
    """
    Va = np.array(mesh_a.vertices)
    Vb = np.array(mesh_b.vertices)
    Ha, _ = mean_curvature(mesh_a)
    Hb, _ = mean_curvature(mesh_b)

    # Transfer Hb onto mesh_a's vertex positions
    _, nn_idx = cKDTree(Vb).query(Va)
    Hb_on_a = Hb[nn_idx.astype(int)]

    # Interior only
    mask = _interior_mask(Va)
    ha = Ha[mask]
    hb = Hb_on_a[mask]

    ha_c = ha - ha.mean()
    hb_c = hb - hb.mean()
    denom = np.sqrt((ha_c**2).sum() * (hb_c**2).sum()) + 1e-12
    return float(np.dot(ha_c, hb_c) / denom)


def compare_meshes(mesh_a: trimesh.Trimesh,
                   mesh_b: trimesh.Trimesh,
                   names: tuple = ('Mesh A', 'Mesh B'),
                   print_results: bool = True) -> dict:
    """
    Full pairwise comparison. The three numbers that matter:
      wrinkle_chamfer  : do wrinkle LOCATIONS match?  (lower = better)
      curvature_corr   : do curvature PATTERNS match? (higher = better)
      full_chamfer     : overall geometric deviation
    """
    wc = wrinkle_chamfer(mesh_a, mesh_b)
    cc = curvature_correlation(mesh_a, mesh_b)
    Ha, _ = mean_curvature(mesh_a)
    Hb, _ = mean_curvature(mesh_b)

    results = {
        'wrinkle_chamfer':   wc['wrinkle_chamfer'],
        'full_chamfer':      wc['full_chamfer'],
        'curvature_corr':    cc,
        f'p90_H_{names[0]}': float(np.percentile(Ha, 90)),
        f'p90_H_{names[1]}': float(np.percentile(Hb, 90)),
    }

    if print_results:
        print(f"\nComparison: {names[0]} vs {names[1]}")
        print(f"  Wrinkle Chamfer (location consistency) : "
              f"{wc['wrinkle_chamfer']:.6f}   lower = wrinkles in same place")
        print(f"  Curvature correlation                  : "
              f"{cc:.4f}             higher = same pattern")
        print(f"  Full Chamfer                           : "
              f"{wc['full_chamfer']:.6f}")
        print(f"  p90|H|  {names[0]}: {float(np.percentile(Ha,90)):.4f}   "
              f"{names[1]}: {float(np.percentile(Hb,90)):.4f}")

    return results

def main():
    parser = argparse.ArgumentParser(description="Compare simulated garment meshes against a reference mesh.")
    parser.add_argument("--reference", required=True, help="reference OBJ path")
    parser.add_argument("--meshes", required=True, nargs="+", help="OBJ files to compare")
    parser.add_argument("--output", help="optional JSON file for metric results")
    args = parser.parse_args()

    ref_mesh = trimesh.load(args.reference, process=False)
    all_results = {}
    for mesh_path in args.meshes:
        print(f"\nMesh: {mesh_path}")
        mesh = trimesh.load(mesh_path, process=False)
        all_results[mesh_path] = compare_meshes(ref_mesh, mesh, names=("reference", mesh_path))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved metrics to {args.output}")


if __name__ == "__main__":
    main()
