import math
import torch
import numpy as np
from torch.utils.data import Dataset
from .config import device

# Parts of the code (sample_points_from_meshes) borrowed from Meta Platforms, Inc.
from typing import Tuple, Union
try:
    from pytorch3d.structures import Meshes
    from pytorch3d.ops.mesh_face_areas_normals import mesh_face_areas_normals
    from pytorch3d.ops.packed_to_padded import packed_to_padded
except ImportError:
    Meshes = None
    mesh_face_areas_normals = None
    packed_to_padded = None
from typing import NamedTuple

def sample_points_from_meshes(
    meshes,
    num_samples: int = 10000,
    return_normals: bool = False
) -> Union[
    torch.Tensor,
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]:
    """
    Convert a batch of meshes to a batch of pointclouds by uniformly sampling
    points on the surface of the mesh with probability proportional to the
    face area.

    Args:
        meshes: A Meshes object with a batch of N meshes.
        num_samples: Integer giving the number of point samples per mesh.

    Returns:
        3-element tuple containing

        - **samples_xyz**: FloatTensor of shape (N, num_samples, 3) giving the
          coordinates of sampled points for each mesh in the batch. For empty
          meshes the corresponding row in the samples array will be filled with 0.
        - **samples_uv**: FloatTensor of shape (N, num_samples, 2) giving the uv coordinates
          for each sampled point. For empty meshes the corresponding row in the array will
          be filled with 0.
    """
    if mesh_face_areas_normals is None or packed_to_padded is None:
        raise ImportError("Mesh sampling requires PyTorch3D. Install the optional geometry dependencies.")

    if meshes.isempty():
        raise ValueError("Meshes are empty.")

    verts = meshes.verts_packed()
    if not torch.isfinite(verts).all():
        raise ValueError("Meshes contain nan or inf.")

    faces = meshes.faces_packed()
    mesh_to_face = meshes.mesh_to_faces_packed_first_idx()
    num_meshes = len(meshes)
    num_valid_meshes = torch.sum(meshes.valid)  # Non empty meshes.

    # Initialize samples tensor with fill value 0 for empty meshes.
    samples_xyz = torch.zeros((num_meshes, num_samples, 3), device=meshes.device)

    # Only compute samples for non empty meshes
    with torch.no_grad():
        areas, face_normals = mesh_face_areas_normals(verts, faces)  # Face areas can be zero.
        max_faces = meshes.num_faces_per_mesh().max().item()
        areas_padded = packed_to_padded(
            areas, mesh_to_face[meshes.valid], max_faces
        )  # (N, F)

        sample_face_idxs = areas_padded.multinomial(
            num_samples, replacement=True
        )  # (N, num_samples)
        sample_face_idxs += mesh_to_face[meshes.valid].view(num_valid_meshes, 1)

    # Get the vertex coordinates of the sampled faces.
    face_verts = verts[faces]
    v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]

    # Randomly generate barycentric coords.
    w0, w1, w2 = _rand_barycentric_coords(
        num_valid_meshes, num_samples, verts.dtype, verts.device
    )

    # Use the barycentric coords to get a point on each sampled face.
    a = v0[sample_face_idxs]  # (N, num_samples, 3)
    b = v1[sample_face_idxs]
    c = v2[sample_face_idxs]
    samples_xyz[meshes.valid] = w0[:, :, None] * a + w1[:, :, None] * b + w2[:, :, None] * c

    samples_uv = torch.zeros((num_meshes, num_samples, 2), device=meshes.device)
    
    if return_normals:
        sample_normals = torch.zeros((num_meshes, num_samples, 3), device=meshes.device)
        sample_normals[meshes.valid] = face_normals[sample_face_idxs]
        return samples_xyz, samples_uv, sample_normals
    else:
        return samples_xyz, samples_uv

def _rand_barycentric_coords(
    size1, size2, dtype: torch.dtype, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Helper function to generate random barycentric coordinates which are uniformly
    distributed over a triangle.

    Args:
        size1, size2: The number of coordinates generated will be size1*size2.
                      Output tensors will each be of shape (size1, size2).
        dtype: Datatype to generate.
        device: A torch.device object on which the outputs will be allocated.

    Returns:
        w0, w1, w2: Tensors of shape (size1, size2) giving random barycentric
            coordinates
    """
    uv = torch.rand(2, size1, size2, dtype=dtype, device=device)
    u, v = uv[0], uv[1]
    u_sqrt = u.sqrt()
    w0 = 1.0 - u_sqrt
    w1 = u_sqrt * (1.0 - v)
    w2 = u_sqrt * v
    return w0, w1, w2

def get_mgrid(sidelen: Union[Tuple[int], Tuple[int, int]], stratified=False, dim=2):
    # Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    if dim == 1:
        sidelen_x = sidelen[0] if stratified else sidelen[0] - 1
        grid_coords = np.stack(np.mgrid[:sidelen[0]], axis=-1)[None, ..., None].astype(np.float32)
        grid_coords[..., 0] = grid_coords[..., 0] / sidelen_x
    elif dim == 2:
        grid_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1]], axis=-1)[None, ...].astype(np.float32)
        if stratified: 
            sidelen_x, sidelen_y = sidelen[0], sidelen[1]
        else: 
            sidelen_x, sidelen_y = sidelen[0] - 1, sidelen[1] - 1
        grid_coords[0, :, :, 0] = grid_coords[0, :, :, 0] / sidelen_x
        grid_coords[0, :, :, 1] = grid_coords[0, :, :, 1] / sidelen_y
    else:
        raise NotImplementedError('Not implemented for dim=%d' % dim)
    grid_coords = torch.Tensor(grid_coords).to(device).view(-1, dim)
    return grid_coords

class DomainBounds(NamedTuple):
    u_max: float
    v_max: float
    
class Sampler(Dataset):
    def __init__(self, n_spatial_samples: int):
        self.n_spatial_samples = n_spatial_samples
    
    def __len__(self):
        return 1

class SamplerTemporal(Dataset):
    def __init__(self, n_spatial_samples: int, n_temporal_samples: int):
        self.n_spatial_samples = n_spatial_samples
        self.n_temporal_samples = n_temporal_samples
        
        self.cell_temporal_coords = get_mgrid((self.n_temporal_samples,), stratified=True, dim=1)
    
    def __len__(self):
        return 1
    
    def get_temporal_coords(self):
        temporal_coords = self.cell_temporal_coords.clone()
        # Add random initialised values to temporal domain.
        t_rand_temporal = torch.rand([self.n_temporal_samples, 1], device=device) / self.n_temporal_samples
        temporal_coords += t_rand_temporal
        temporal_coords.requires_grad_(True)    
        temporal_coords = temporal_coords.repeat_interleave(self.n_spatial_samples, 0)  
        return temporal_coords
                           
class GridSampler(Sampler):
    def __init__(self, n_spatial_samples: int, domain_bounds: DomainBounds):
        super().__init__(n_spatial_samples)
                
        self.domain_bounds = domain_bounds
        self.spatial_sidelen = math.isqrt(n_spatial_samples)
        
        self.cell_rest_points = get_mgrid((self.spatial_sidelen, self.spatial_sidelen), stratified=True, dim=2)            

    def __getitem__(self, idx):    
        if idx > 0: raise IndexError
         
        rest_points = self.cell_rest_points.clone()        
        
        t_rand_spatial = torch.rand([self.n_spatial_samples, 2], device=device) / self.spatial_sidelen 
        # Add random initialised values in spatial domain.
        rest_points += t_rand_spatial
        
        rest_points[...,0] *= self.domain_bounds.u_max
        rest_points[...,1] *= self.domain_bounds.v_max
        rest_points.requires_grad_(True)
      
        return rest_points #, self.get_temporal_coords()            

class MeshSampler(Sampler):
    def __init__(self, n_spatial_samples: int, reference_mesh: "Meshes"):
        if Meshes is None:
            raise ImportError("MeshSampler requires PyTorch3D. Install the optional geometry dependencies.")
        super().__init__(n_spatial_samples)
        
        self.verts = reference_mesh.verts_packed()
        self.faces = reference_mesh.faces_packed()
        
        self.face_areas = reference_mesh.faces_areas_packed()
        
        self.num_faces = self.faces.shape[0]
            
    def __getitem__(self, idx):    
        face_indices = torch.randint(0, self.num_faces, (self.n_spatial_samples,), device=self.verts.device)

        sample_faces = self.faces[face_indices]
        v0 = self.verts[sample_faces[:, 0]]
        v1 = self.verts[sample_faces[:, 1]]
        v2 = self.verts[sample_faces[:, 2]]

        # Standard Monte Carlo
        weights = self.face_areas[face_indices] * self.num_faces

        r1 = torch.rand(self.n_spatial_samples, 1, device=self.verts.device)
        r2 = torch.rand(self.n_spatial_samples, 1, device=self.verts.device)
        
        sqrt_r1 = torch.sqrt(r1)
        u = 1.0 - sqrt_r1
        v = sqrt_r1 * (1.0 - r2)
        w = sqrt_r1 * r2

        rest_points = u * v0 + v * v1 + w * v2
        rest_points.requires_grad_(True)
                    
        return rest_points, weights.view(-1)

class SDFSampler(Sampler):
    def __init__(self, n_spatial_samples: int, reference_mlp, bbox_min=-1.0, bbox_max=1.0, sdf_threshold=1e-3, max_trials=20):
        super().__init__(n_spatial_samples)
        self.reference_mlp = reference_mlp.to(device)
        self.bbox_min = bbox_min
        self.bbox_max = bbox_max
        self.sdf_threshold = sdf_threshold
        self.max_trials = max_trials

    def __getitem__(self, idx):
        if idx > 0:
            raise IndexError

        n_target = self.n_spatial_samples
        all_points = []
        trials = 0

        while len(all_points) < n_target and trials < self.max_trials:
            x = torch.empty((n_target * 4, 3), device=device).uniform_(self.bbox_min, self.bbox_max)
            x.requires_grad_(True)

            sdf_vals = self.reference_mlp(x).squeeze()

            mask = sdf_vals.abs() < self.sdf_threshold
            near_surface = x[mask]
            all_points.append(near_surface.detach())

            trials += 1

        if len(all_points) == 0:
            raise RuntimeError("Failed to sample near-surface points from SDF. Try increasing sdf_threshold or max_trials.")

        rest_points = torch.cat(all_points, dim=0)[:n_target]
        rest_points.requires_grad_(True)

        return rest_points


class UDFSampler(Sampler):
    def __init__(
        self, 
        n_spatial_samples: int, 
        reference_mlp,
        bbox_min=-1.0, 
        bbox_max=1.0, 
        udf_threshold=0.01,
        max_trials=20,
        oversample_factor=10,
    ):
        super().__init__(n_spatial_samples)
        self.reference_mlp = reference_mlp.to(device)
        self.bbox_min = bbox_min
        self.bbox_max = bbox_max
        self.udf_threshold = udf_threshold
        self.max_trials = max_trials
        self.oversample_factor = oversample_factor

    def __getitem__(self, idx):
        if idx > 0:
            raise IndexError

        n_target = self.n_spatial_samples
        all_points = []
        trials = 0

        while len(all_points) < n_target and trials < self.max_trials:
            n_candidates = n_target * self.oversample_factor
            x = torch.empty(
                (n_candidates, 3), 
                device=device
            ).uniform_(self.bbox_min, self.bbox_max)

            with torch.no_grad():
                udf_vals = self.reference_mlp(x)['model_out'].squeeze()

            mask = udf_vals < self.udf_threshold
            near_surface = x[mask].detach()
            
            if len(near_surface) > 0:
                all_points.append(near_surface)

            trials += 1

        rest_points = torch.cat(all_points, dim=0)
        
        if len(rest_points) < n_target:
            remaining = n_target - len(rest_points)
            
            extra_candidates = torch.empty((remaining * 5, 3), device=device).uniform_(
                self.bbox_min, self.bbox_max
            )
            with torch.no_grad():
                extra_udf = self.reference_mlp(extra_candidates)['model_out'].squeeze()
            _, closest_indices = torch.topk(-extra_udf, min(remaining, len(extra_udf)))
            extra = extra_candidates[closest_indices]
            
            rest_points = torch.cat([rest_points, extra], dim=0)
        else:
            indices = torch.randperm(len(rest_points))[:n_target]
            rest_points = rest_points[indices]

        rest_points.requires_grad_(True)
        return rest_points
