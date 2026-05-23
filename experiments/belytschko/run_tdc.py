from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import os
from tqdm import trange
import numpy as np
import random
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

seed=42
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from typing import NamedTuple
from torch.utils.tensorboard import SummaryWriter  
from garment_tdc.sampling import get_mgrid
from garment_tdc.differential_operators import jacobian
from pytorch3d.io import save_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh.textures import TexturesUV
from garment_tdc.config import device
from torch.nn.functional import normalize
import matplotlib.pyplot as plt


class DomainBounds(NamedTuple):
    u_max: float
    v_max: float

class RestSurface():
    def __init__(self, geometry_name: str, test_n_spatial_samples: int, domain_bounds: DomainBounds):
        self.geometry_name = geometry_name
        
        test_spatial_sidelen = math.isqrt(test_n_spatial_samples)
        rest_points = get_mgrid((test_spatial_sidelen, test_spatial_sidelen), stratified=False, dim=2)[None]
            
        rest_points[...,0] *= domain_bounds.u_max
        rest_points[...,1] *= domain_bounds.v_max
        
        vertices = self(rest_points)[0]
        faces = torch.tensor(generate_mesh_topology(test_spatial_sidelen), device=device)
        texture = TexturesUV(maps=torch.empty(1, 1, 1, 1, device=device), faces_uvs=[faces], verts_uvs=rest_points)
        self.template_mesh = Meshes(verts=[vertices], faces=[faces], textures=texture).to(device)
        
    def __call__(self, rest_points: torch.Tensor) -> torch.Tensor:
        xi__1 = rest_points[...,0] 
        xi__2 = rest_points[...,1]
        
        match self.geometry_name:
            case 'square_plate' | 'napkin':
                midsurface_positions = torch.stack([xi__1, xi__2, 0.*(xi__1**2 - xi__2**2)], dim=2)
            case 'scordelis_lo_roof':
                R = 25.0
                midsurface_positions = torch.stack([R * torch.cos(xi__1 + 5 * math.pi / 18), R * torch.sin(xi__1 + 5 * math.pi / 18), xi__2], dim=2)
            case 'pinched_cylinder_rigid_diaphragms':
                R = 300.0
                midsurface_positions = torch.stack([
                    R * torch.cos(xi__1), 
                    xi__2,
                    R * torch.sin(xi__1)
                ], dim=2)            
            case 'compressed_cylinder' | 'twisted_cylinder':
                R = 0.25
                midsurface_positions = torch.stack([
                    R * torch.cos(xi__1), 
                    xi__2, 
                    R * torch.sin(xi__1)
                ], dim=2)                
            case 'hemispherical_shell':
                R = 10
                midsurface_positions = torch.stack([
                    R * torch.sin(xi__2) * torch.cos(xi__1), 
                    R * torch.sin(xi__2) * torch.sin(xi__1), 
                    R * torch.cos(xi__2)
                ], dim=2)
                
        return midsurface_positions

class GridSampler(Dataset):
    def __init__(self, n_spatial_samples: int, domain_bounds: DomainBounds, geometry_name: str):
        self.n_spatial_samples = n_spatial_samples
        self.domain_bounds = domain_bounds
        self.geometry_name = geometry_name
        
        self.spatial_sidelen = math.isqrt(n_spatial_samples)
        self.cell_rest_points = get_mgrid((self.spatial_sidelen, self.spatial_sidelen), stratified=True, dim=2)            

    def __len__(self):
        return 1
    
    def __getitem__(self, idx):    
        if idx > 0: raise IndexError
         
        rest_points = self.cell_rest_points.clone()
        
        t_rand_spatial = torch.rand([rest_points.shape[0], 2], device=device) / self.spatial_sidelen        
        rest_points += t_rand_spatial
        
        rest_points[...,0] *= self.domain_bounds.u_max
        rest_points[...,1] *= self.domain_bounds.v_max
        rest_points.requires_grad_(True)

        return rest_points
    
class Boundary:
    def __init__(self, geometry_name: str, boundary_condition_name: str, domain_bounds: DomainBounds):
        self.geometry_name = geometry_name
        self.boundary_condition_name = boundary_condition_name
        self.domain_bounds = domain_bounds
        self.boundary_support = 0.01
    
    def normalization(self, points: torch.Tensor) -> torch.Tensor:
        # Normalize points to [0, 1] range
        if self.geometry_name in ['napkin', 'square_plate']:
            normalized_coords = torch.cat([
                (points[..., 0:1]) / self.domain_bounds.u_max, 
                (points[..., 1:2]) / self.domain_bounds.v_max,
                points[..., 2:3]  # Keep z-coordinate as is for square plate (should be 0)
            ], dim=2)
            return normalized_coords
        elif self.geometry_name in ['scordelis_lo_roof']:
            normalized_coords = torch.cat([
                ((points[..., 0:1]) / 25.0 + 1) / 2, 
                ((points[..., 1:2]) / 25.0 + 1) / 2,
                (points[..., 2:3]) / self.domain_bounds.v_max,
            ], dim=2)
            return normalized_coords
        elif self.geometry_name in ['pinched_cylinder_rigid_diaphragms']:
            R = 300.0
    
            xi1 = torch.atan2(points[..., 2:3], points[..., 0:1])
            xi2 = points[..., 1:2]
            
            normalized_coords = torch.cat([
                (torch.cos(xi1) + 1)/2,
                (torch.sin(xi1) + 1)/2,
                xi2/self.domain_bounds.v_max
            ], dim=2)
            return normalized_coords

    def dirichlet_condition(self, deformations: torch.Tensor, points: torch.Tensor) -> torch.Tensor:        
        min_coords, _ = torch.min(points, dim=1, keepdim=True)
        max_coords, _ = torch.max(points, dim=1, keepdim=True)
        
        x_min, y_min, z_min = min_coords.split(1, dim=-1)
        x_max, y_max, z_max = max_coords.split(1, dim=-1)
        
        x, y, z = points.split(1, dim=-1)
        match self.boundary_condition_name:
            case 'top_left_fixed':
                corner_point = torch.cat([x_min, y_max, z_max], dim=-1)
                corner_mask = torch.exp(-torch.sum((points - corner_point)**2, dim=-1, keepdim=True) / self.boundary_support)
                deformations = deformations * (1 - corner_mask)
            case 'square_plate_clamped':
                left_edge = points[...,0:1] / self.domain_bounds.u_max
                right_edge = points[...,1:2] / self.domain_bounds.v_max
                edge_mask = left_edge * right_edge * (1 - left_edge) * (1 - right_edge)
                
                deformations = deformations * edge_mask
            case 'scordelis_lo_roof_boundaries':
                bottom_end = points[..., 2:3] / self.domain_bounds.v_max
                top_end = (self.domain_bounds.v_max - points[..., 2:3]) / self.domain_bounds.v_max
                end_mask = bottom_end * top_end
                deformations = torch.cat([
                    deformations[..., 0:1] * end_mask,
                    deformations[..., 1:2] * end_mask,
                    deformations[..., 2:3]
                ], dim=2)
            case 'pinched_cylinder_rigid_diaphragms_boundaries':
                bottom_end = points[..., 1:2] / self.domain_bounds.v_max
                top_end = (self.domain_bounds.v_max - points[..., 1:2]) / self.domain_bounds.v_max
                end_mask = 4.0 * bottom_end * top_end
                deformations = torch.cat([
                    deformations[...,0:1] * end_mask,
                    deformations[...,1:2],
                    deformations[...,2:3] * end_mask
                ], dim=2)
        return deformations

class SineLayer(nn.Module):      
    def __init__(self, in_features: int, out_features: int, bias=True, is_first=False, omega_0=30.):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
   
class SirenDeformationField(nn.Module):
    def __init__(self, boundary: Boundary, in_features=3, hidden_features=512, hidden_layers=5, out_features=3, outermost_linear=True, first_omega_0=30., hidden_omega_0=30.):
        super().__init__()
        self.boundary = boundary
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, np.sqrt(6 / hidden_features) / hidden_omega_0)
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, is_first=False, omega_0=hidden_omega_0))    
        self.net = nn.Sequential(*self.net)
    
    def forward(self, points: torch.Tensor) -> torch.Tensor:                
        
        norm_points = self.boundary.normalization(points)
        deformations = self.net(norm_points)
        deformations = self.boundary.dirichlet_condition(deformations, points)

        return deformations

def generate_mesh_topology(spatial_sidelen):
    rows = cols = spatial_sidelen
    last_face_index = cols * (rows - 1)
    
    first_face_bl = [0, cols, 1]  
    first_face_tr = [cols + 1, 1, cols]  
    all_faces = []
    for first_face in [first_face_bl, first_face_tr]:
        last_face = [i + last_face_index - 1 for i in first_face]
        faces = np.linspace(first_face, last_face, last_face_index)
        faces = np.reshape(faces, (rows - 1, cols, 3))
        faces = np.delete(faces, cols - 1, 1)
        faces = np.reshape(faces, (-1, 3))   
        all_faces.append(faces)
    return np.concatenate(all_faces, axis=0)

def save_points(verts, indices, directory, name, num_verts):
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    for i in indices:
        filename = os.path.join(directory, f"points_{name}_{i}.txt")
        np.savetxt(filename, verts[i, :num_verts].detach().cpu().numpy())

def plot_displacement(coords, displacements, component_idx, directory, step, sidelen, geometry_name):
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    coords_np = coords.detach().cpu().numpy().squeeze()
    displacements_np = displacements.detach().cpu().numpy().squeeze()
    
    if geometry_name in ['square_plate', 'napkin']:
        component_name = 'Z'
        title = f'Square Plate: Z-Displacement (ū³) at Step {step}'
    elif geometry_name == 'scordelis_lo_roof':
        component_name = 'Y'
        title = f'Scordelis-Lo Roof: Y-Displacement (ū²) at Step {step}'
    elif geometry_name in ['pinched_cylinder_rigid_diaphragms']:
        component_name = 'Z'
        if geometry_name == 'pinched_cylinder_rigid_diaphragms':
            title = f'Pinched Cylinder (Fixed): Z-Displacement (ū³) at Step {step}'
        else:
            title = f'Pinched Cylinder (Free): Z-Displacement (ū³) at Step {step}'
    else:
        component_map = {0: 'X', 1: 'Y', 2: 'Z'}
        component_name = component_map.get(component_idx, f'Component {component_idx}')
        title = f'Displacement ({component_name}) at Step {step}'

    plt.figure(figsize=(10, 8))
    plt.title(title)
    
    if coords_np.shape[0] == sidelen * sidelen:
        x = coords_np[:, 0].reshape(sidelen, sidelen)
        y = coords_np[:, 1].reshape(sidelen, sidelen)
        
        if geometry_name in ['pinched_cylinder_rigid_diaphragms']:
            x = np.unwrap(x, period=2*np.pi)
        
        z = displacements_np[:, component_idx].reshape(sidelen, sidelen)
        
        # Create contour plot
        contour = plt.contourf(x, y, z, levels=50, cmap='inferno')
        plt.colorbar(contour, label=f'{component_name} Displacement')
        
        if geometry_name in ['square_plate', 'napkin']:
            plt.xlabel('ξ¹')
            plt.ylabel('ξ²')
        elif geometry_name == 'scordelis_lo_roof':
            plt.xlabel('ξ¹ (angle)')
            plt.ylabel('ξ² (length)')
        elif geometry_name in ['pinched_cylinder_rigid_diaphragms']:
            plt.xlabel('ξ¹ (angle)')
            plt.ylabel('ξ² (length)')
    else:
        plt.scatter(coords_np[:, 0], coords_np[:, 1], c=displacements_np[:, component_idx], cmap='inferno')
        plt.colorbar(label=f'{component_name} Displacement')
        plt.xlabel('Rest coordinate 1')
        plt.ylabel('Rest coordinate 2')

    plt.tight_layout()
    plt.savefig(os.path.join(directory, f"displacement_{geometry_name}_{component_name}_{step:06d}.png"), dpi=300, bbox_inches='tight')
    plt.close()

loss_steps = []
loss_values = []

def track_loss(step, loss, directory, geometry_name, lower_percentile=5, upper_percentile=95):
    loss_steps.append(step)
    loss_values.append(loss)
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    if len(loss_values) > 10:
        lower_limit = np.percentile(loss_values, lower_percentile)
        upper_limit = np.percentile(loss_values, upper_percentile)
    else:
        lower_limit = min(loss_values)
        upper_limit = max(loss_values)
    
    margin = 0.1 * (upper_limit - lower_limit if upper_limit > lower_limit else abs(upper_limit))
    lower_limit -= margin
    upper_limit += margin
    
    plt.figure(figsize=(10, 6))
    plt.plot(loss_steps, loss_values, 'b-', label='Loss')
    
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title(f'Loss History for {geometry_name}')
    plt.legend()
    plt.grid(True)

    plt.ylim(bottom=lower_limit, top=upper_limit)
    
    filename = os.path.join(directory, f"loss_history_{geometry_name}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


mean_def_steps = []
mean_def_values = []

def track_mean_deformation(step, mean_def, directory, geometry_name):
    mean_def_steps.append(step)
    mean_def_values.append(mean_def)
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    plt.figure(figsize=(10, 6))
    plt.plot(mean_def_steps, mean_def_values, 'g-', label='Mean Deformation')
    
    plt.xlabel('Iteration')
    plt.ylabel('Mean Deformation')
    plt.title(f'Mean Deformation History for {geometry_name}')
    plt.legend()
    plt.grid(True)
    
    filename = os.path.join(directory, f"mean_deformation_history_{geometry_name}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

displacement_steps = []
displacement_values = []

import csv
from datetime import datetime, timedelta

# This will be set at training start
training_start_time = None
csv_log_file = None

def track_displacement(step, max_disp, directory, geometry_name, reference_value=None):
    global csv_log_file

    displacement_steps.append(step)
    displacement_values.append(max_disp)
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    plt.figure(figsize=(10, 6))
    plt.plot(displacement_steps, displacement_values, 'b-', label='Max Displacement')
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))  # scientific notation
    
    # Max value
    max_val = max(displacement_values)
    max_step = displacement_steps[displacement_values.index(max_val)]
    plt.scatter(max_step, max_val, color='red', marker='o', label=f'Max: {max_val:.3e}')
    plt.annotate(f'{max_val:.3e}', (max_step, max_val),
                 textcoords="offset points", xytext=(10, 5), ha='left', color='red')
    
    # Last value
    last_val = displacement_values[-1]
    last_step = displacement_steps[-1]
    plt.scatter(last_step, last_val, color='green', marker='x', label=f'Last: {last_val:.3e}')
    plt.annotate(f'{last_val:.3e}', (last_step, last_val),
                 textcoords="offset points", xytext=(10, -15), ha='left', color='green')
    
    if reference_value is not None:
        plt.axhline(y=reference_value, color='r', linestyle='--', label=f'Reference: {reference_value:.3e}')
    
    plt.xlabel('Iteration')
    plt.ylabel('Max Displacement')
    plt.title(f'Max Displacement History for {geometry_name}')
    plt.legend()
    plt.grid(True)
    
    filename = os.path.join(directory, f"max_displacement_history_{geometry_name}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    # =========================
    # Logging to new timestamped CSV
    # =========================
    if csv_log_file is None:
        timestamp = training_start_time.strftime("%Y%m%d_%H%M%S")
        csv_log_file = os.path.join(directory, f"displacement_log_{geometry_name}_{timestamp}.csv")
        # Write header once
        with open(csv_log_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "elapsed_time_hh:mm:ss", "wall_clock_end",
                "step", "max_disp_current",
                "max_val_overall", "max_step_overall",
                "last_val", "last_step",
                "reference_value"
            ])

    elapsed = datetime.now() - training_start_time
    elapsed_str = str(timedelta(seconds=int(elapsed.total_seconds())))
    
    with open(csv_log_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            elapsed_str, datetime.now().isoformat(),
            step, f"{max_disp:.6e}",
            f"{max_val:.6e}", max_step,
            f"{last_val:.6e}", last_step,
            f"{reference_value:.6e}" if reference_value is not None else None
        ])

class SurfaceGeometry(): 
    def __init__(self, geometry_name: str, rest_surface: RestSurface, n_spatial_samples: int):
        self.n_spatial_samples = n_spatial_samples 
        self.rest_surface = rest_surface
        self.geometry_name = geometry_name

    def __call__(self, rest_points: torch.Tensor):
        self.rest_points = rest_points

        self.points = self.rest_surface(self.rest_points)
        self.points.requires_grad_(True)
        batch_size, n_points, _ = self.points.shape

        match self.geometry_name:
            case 'square_plate' | 'napkin':
                self.normals = torch.tensor([0., 0., 1.], device=device).view(1, 1, 3).expand(batch_size, n_points, -1)
                self.a_1 = torch.tensor([1., 0., 0.], device=device).view(1, 1, 3).expand(batch_size, n_points, -1)
                self.a_2 = torch.tensor([0., 1., 0.], device=device).view(1, 1, 3).expand(batch_size, n_points, -1)
                self.weingarten = torch.zeros((batch_size, n_points, 3, 3), device=device)
            case 'scordelis_lo_roof':
                R = 25.0
                n_xy = torch.cat([self.points[..., 0:1] / R, self.points[..., 1:2] / R, torch.zeros((batch_size, n_points, 1), device=device)], dim=-1)
                self.normals = torch.nn.functional.normalize(n_xy, dim=2, eps=1e-12)

                e1 = torch.cat([-self.points[..., 1:2] / R, self.points[..., 0:1] / R, torch.zeros((batch_size, n_points, 1), device=device)], dim=-1)
                e1 = torch.nn.functional.normalize(e1, dim=2, eps=1e-12)

                e2 = torch.tensor([0., 0., 1.], device=device).view(1, 1, 3).expand(batch_size, n_points, 3)

                self.a_1 = e1
                self.a_2 = e2

                kappa = 1.0 / R
                self.weingarten = kappa * torch.einsum('bni,bnj->bnij', e1, e1)
            case 'pinched_cylinder_rigid_diaphragms':
                R = 300.0
                self.normals = torch.cat([self.points[..., 0:1]/R,
                              torch.zeros_like(self.points[..., 0:1]),
                              self.points[..., 2:3]/R], dim=-1)
                self.normals = torch.nn.functional.normalize(self.normals, dim=2, eps=1e-12)
                
                e1 = torch.cat([-self.points[..., 2:3],
                    torch.zeros_like(self.points[..., 0:1]),
                    self.points[..., 0:1]], dim=-1)
                e2 = torch.tensor([0., 1., 0.], device=device).view(1, 1, 3).expand(batch_size, n_points, -1)
                self.a_1, self.a_2 = e1, e2

                e1_unit = torch.nn.functional.normalize(e1, dim=2, eps=1e-12)
                self.weingarten = (1.0 / R) * torch.einsum('bni,bnj->bnij', e1_unit, e1_unit)

        a_1_1 = torch.einsum('ijk,ijk->ij', self.a_1, self.a_1)
        a_1_2 = torch.einsum('ijk,ijk->ij', self.a_1, self.a_2)
        a_2_2 = torch.einsum('ijk,ijk->ij', self.a_2, self.a_2)
        det_g = a_1_1 * a_2_2 - a_1_2 * a_1_2
        self.area_element = torch.sqrt(det_g.clamp_min(1e-12))
    
class Strain(NamedTuple):
    epsilon_M: torch.Tensor
    epsilon_B: torch.Tensor

def tangential_gradient(field: torch.Tensor, ref_geometry: SurfaceGeometry, projector: torch.Tensor, is_scalar=False):
    if is_scalar:
        if field.ndim == 2:
            field = field.unsqueeze(-1)
        elif field.ndim == 3 and field.shape[-1] != 1:
            raise ValueError("Expected scalar field to have shape [B, P] or [B, P, 1]")
    
    grad_u_tilde, status = jacobian(field, ref_geometry.points)

    if status != 0:
        raise ValueError("Obtained NaN values during gradient calculation.")

    if is_scalar:
        grad_u_tilde = grad_u_tilde.squeeze(-2)
        tangential_grad = torch.einsum('bpi,bpij->bpj', grad_u_tilde, projector)
    else:
        tangential_grad = torch.einsum('bpik,bpkj->bpij', grad_u_tilde, projector)

    return tangential_grad

def compute_strain(deformations: torch.Tensor, ref_geometry: SurfaceGeometry, i: int, nonlinear_strain=False): 
    batch_size, num_points, _ = deformations.shape

    identity = torch.eye(3, device=device).expand(batch_size, num_points, 3, 3)
    n_dyad = torch.einsum('bpi,bpj->bpij', ref_geometry.normals, ref_geometry.normals)
    projector = identity - n_dyad

    grad_u = tangential_gradient(deformations, ref_geometry, projector)
    grad_uT = grad_u.transpose(-1, -2)
    
    epsilon_M_linear = 0.5 * (grad_u + grad_uT)
    
    H = ref_geometry.weingarten
    u_dot_n = torch.einsum('bpi,bpi->bp', deformations, ref_geometry.normals)
    w = torch.einsum('bpij,bpj->bpi', H, deformations) - tangential_gradient(
        u_dot_n, ref_geometry, projector, is_scalar=True
    )

    grad_w = tangential_gradient(w, ref_geometry, projector)

    # Parameterisation-free curvature correction.
    # `w` is the first-order change of the unit normal.  The bending strain is
    # not simply sym(grad_w): on a curved surface that incorrectly penalizes
    # rigid rotations.  The correction below subtracts the rotation of the
    # reference curvature tensor induced by the displacement gradient.  It is
    # zero for flat plates because H == 0, which is why the square-plate case can
    # look correct even when curved shells are wrong.
    grad_w_corrected = grad_w - torch.einsum('bpik,bpkj->bpij', grad_u, H)
    grad_w_correctedT = grad_w_corrected.transpose(-1, -2)

    epsilon_B_linear = 0.5 * (grad_w_corrected + grad_w_correctedT)

    if nonlinear_strain:
        epsilon_M_nonlinear = 0.5 * torch.einsum('bpki,bpkj->bpij', grad_u, grad_u)
        Hw = torch.einsum('bpij,bpj->bpi', H, w)
        epsilon_B_nonlinear = 0.5 * (
            torch.einsum('bpi,bpj->bpij', Hw, w) + 
            torch.einsum('bpj,bpi->bpij', w, Hw)
        )
        
        epsilon_M = epsilon_M_linear + epsilon_M_nonlinear
        epsilon_B = epsilon_B_linear + epsilon_B_nonlinear
    else: 
        epsilon_M = epsilon_M_linear
        epsilon_B = epsilon_B_linear
    
    epsilon_M = torch.einsum('bpik,bpkl,bplj->bpij', projector, epsilon_M, projector)
    epsilon_B = torch.einsum('bpik,bpkl,bplj->bpij', projector, epsilon_B, projector)
    # epsilon_B = torch.zeros_like(epsilon_B)
    
    return Strain(epsilon_M, epsilon_B)
    
class Material():
    def __init__(self, thickness: float, ref_geometry: SurfaceGeometry):
        self.thickness = thickness
        self.ref_geometry = ref_geometry
    
    def compute_internal_energy(self, strain: Strain) -> torch.Tensor:
        raise NotImplementedError

class LinearMaterial(Material):
    def __init__(self, ref_geometry: SurfaceGeometry, youngs_modulus: float, poissons_ratio: float, thickness: float):
        super().__init__(thickness, ref_geometry)
        self.poissons_ratio = poissons_ratio
        self.youngs_modulus = youngs_modulus

        self.mu_ = self.youngs_modulus / (2 * (1 + self.poissons_ratio))
        self.lambda_ = (self.youngs_modulus * self.poissons_ratio) / (1 - self.poissons_ratio**2)
    
    def compute_stress(self, strain: torch.Tensor):
        batch_size, num_points = strain.shape[0:2]
        identity = torch.eye(3, device=strain.device).expand(batch_size, num_points, 3, 3)

        tr_strain = torch.einsum('bpii->bp', strain).unsqueeze(-1).unsqueeze(-1) # [1, 400]

        n_dyad = torch.einsum('bpi,bpj->bpij', self.ref_geometry.normals, self.ref_geometry.normals)
        projector = identity - n_dyad
        stress_3d = 2 * self.mu_ * strain + self.lambda_ * tr_strain * identity
        in_plane_stress = torch.einsum('bpik,bpkl,bplj->bpij', projector, stress_3d, projector)

        # Only adding identity matrix for compatible tensor addition. Will modify once entire codebase is setup
        return in_plane_stress

    def compute_internal_energy(self, strain: Strain) -> torch.Tensor:
        
        sigma_M = self.compute_stress(strain.epsilon_M)
        sigma_B = self.compute_stress(strain.epsilon_B)

        n_tilde = self.thickness * sigma_M
        m_gamma = (self.thickness**3 / 12) * sigma_B

        # Tensor contraction to get energy density
        energy_M = torch.einsum('bpij,bpij->bp', n_tilde, strain.epsilon_M)
        energy_B = torch.einsum('bpij,bpij->bp', m_gamma, strain.epsilon_B)

        return 0.5 * (energy_M + energy_B) # 1

class Energy:
    def __init__(self, ref_geometry: SurfaceGeometry, material: Material, geometry_name: str, domain_bounds: DomainBounds, mass_area_density):
        self.ref_geometry = ref_geometry
        self.material = material
        self.geometry_name = geometry_name
        self.domain_bounds = domain_bounds
        self.get_distributed_load(mass_area_density)
        self.get_point_loads()

    def get_point_loads(self):
        match self.geometry_name:
            case 'pinched_cylinder_rigid_diaphragms':
                coords = torch.tensor([[[0., 300.0, 300.0], 
                                        [0., 300.0, -300.0]]], device=device)
                load_scale = 1.0
                loads = load_scale * torch.tensor([[[0., 0., -1.], [0., 0., 1.]]], device=device)
            case _:
                coords, loads = None, None
                
        self.point_coords, self.point_loads = coords, loads
    
    def get_distributed_load(self, mass_area_density) -> torch.Tensor:
        match self.geometry_name:
            case 'napkin':
                p = [0, -9.8 * mass_area_density, 0]
                load = torch.tensor(p, device=device).expand(1, self.ref_geometry.n_spatial_samples, 3)
            
            case 'square_plate':
                p = [0, 0, -1.0] 
                load = torch.tensor(p, device=device).expand(1, self.ref_geometry.n_spatial_samples, 3)
                
            case 'scordelis_lo_roof':
                p = [0.0, -90.0, 0.0]
                load = torch.tensor(p, device=device).expand(1, self.ref_geometry.n_spatial_samples, 3)
            
            case 'pinched_cylinder_rigid_diaphragms':
                # No distributed load for pinched cylinder - handled via point loads
                p = [0, 0, 0.] 
                load = torch.tensor(p, device=device).expand(1, self.ref_geometry.n_spatial_samples, 3)
                
        self.external_load = load

    def __call__(self, deformations: torch.Tensor, i: int, deformation_field: SirenDeformationField = None) -> torch.Tensor:   
        strain = compute_strain(deformations, self.ref_geometry, i)
        area_element = self.ref_geometry.area_element.unsqueeze(0)
        hyperelastic_strain_energy = self.material.compute_internal_energy(strain)
        
        # Handle point loads for pinched cylinder
        if self.geometry_name in ['pinched_cylinder_rigid_diaphragms']: 
            displacements_load_point = deformation_field(self.point_coords)  # shape [B, n_point_loads, 3]
            ext_energy = -torch.einsum('ijk,ijk->ij', self.point_loads, displacements_load_point)  # shape [B, n_point_loads]

            total_surface_area = 2 * math.pi * 300.0 * 600.0
            internal_energy = total_surface_area * hyperelastic_strain_energy.mean()
            mechanical_energy = internal_energy + ext_energy.sum()
        else:
            # Distributed loads
            external_energy_mid = torch.einsum('ijk,ijk->ij', self.external_load, deformations)
            mechanical_energy = (hyperelastic_strain_energy - external_energy_mid)
            mechanical_energy = torch.sum(mechanical_energy * area_element, dim=-1)
        
        if not i % 100 and tb_writer:
            tb_writer.add_histogram('param/hyperelastic_strain_energy', hyperelastic_strain_energy, i)
        return mechanical_energy

def save_meshes(verts, faces, indices, directory, name, num_verts, verts_uvs=None):
    if not os.path.exists(directory):
        os.makedirs(directory)
    for i in indices:
        mesh_dir = os.path.join(directory, f'{i}')
        os.makedirs(mesh_dir, exist_ok=True)
        
        mesh_verts = verts[i, :num_verts]
        
        filename = os.path.join(mesh_dir, f"{name}_{i:03d}.obj")
        save_obj(
            filename,
            mesh_verts,
            faces,
            verts_uvs=verts_uvs,
            faces_uvs=faces,
        )

def test(deformation_field: SirenDeformationField, rest_surface: RestSurface, i: int, geometry_name: str, ref_value: float):
    test_deformations = deformation_field(rest_surface.template_mesh.verts_padded())
    test_deformed_positions = rest_surface.template_mesh.verts_padded() + test_deformations
    
    tb_writer.add_mesh('simulated_states', test_deformed_positions, faces=rest_surface.template_mesh.textures.faces_uvs_padded(), global_step=i)
    
    if i % 1000 == 0:
        test_meshes_dir = os.path.join('logs_tdc', geometry_name, 'test_meshes')
        plots_dir = os.path.join('logs_tdc', geometry_name, 'plots')
        tracks_dir = os.path.join('logs_tdc', geometry_name)
        
        # save_meshes(test_deformed_positions, rest_surface.template_mesh.textures.faces_uvs_padded()[0], 
        #            [0], test_meshes_dir, f'mesh_{i}', test_deformed_positions.shape[1])
        
        if geometry_name in ['square_plate', 'napkin']:
            plot_displacement(rest_surface.template_mesh.textures.verts_uvs_padded(), test_deformations, 2, os.path.join(plots_dir, 'displacement_plots'), i, 
                            math.isqrt(test_deformed_positions.shape[1]), geometry_name)
        elif geometry_name == 'scordelis_lo_roof':
            plot_displacement(rest_surface.template_mesh.textures.verts_uvs_padded(), test_deformations, 1, os.path.join(plots_dir, 'displacement_plots'), i, 
                            math.isqrt(test_deformed_positions.shape[1]), geometry_name)
        elif geometry_name in ['pinched_cylinder_rigid_diaphragms']:
            plot_displacement(rest_surface.template_mesh.textures.verts_uvs_padded(), test_deformations, 2, os.path.join(plots_dir, 'displacement_plots'), i, 
                            math.isqrt(test_deformed_positions.shape[1]), geometry_name)
        
        # For pinched cylinder cases, check displacement at load point
        if geometry_name in ['pinched_cylinder_rigid_diaphragms']:
            load_coords = torch.tensor([[[0., 300.0, 300.0]]], device=device, requires_grad=True)
            load_point_disp = deformation_field(load_coords)
            max_disp = torch.abs(load_point_disp[0,0,2]).item() * 1e-5
        else:
            max_disp = torch.max(torch.abs(test_deformations)).item()
            
        tb_writer.add_scalar('scalar/max_displacement', max_disp, i)
        print(f'Max displacement at step {i}: {max_disp}')
        track_displacement(i, max_disp, tracks_dir, geometry_name, reference_value=ref_value) 

def train(geometry_name, boundary_condition_name, test_n_spatial_samples, n_iterations, 
          youngs_modulus=5000, poissons_ratio=0.25, thickness=0.0012, mass_area_density=0.144, lrate=5e-6):
    
    log_dir = os.path.join('logs_tdc', geometry_name)
    global tb_writer
    tb_writer = SummaryWriter(log_dir)
    
    # Create directories for visualizations
    test_meshes_dir = os.path.join(log_dir, 'test_meshes')
    plots_dir = os.path.join(log_dir, 'plots')
    os.makedirs(test_meshes_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    lr_decay = 50000
    train_n_spatial_samples = 100
    # Set domain bounds based on geometry
    if geometry_name in ['pinched_cylinder_rigid_diaphragms', 'twisted_cylinder']:
        domain_bounds = DomainBounds(u_max=2 * math.pi, v_max=600.0)  # L = 600 for cylinder
        lr_decay = 20000 # 5000
        train_n_spatial_samples = 10000
        if geometry_name == 'pinched_cylinder_rigid_diaphragms':
            ref_value = 1.825e-5
    elif geometry_name == 'scordelis_lo_roof':
        domain_bounds = DomainBounds(u_max=(4 * math.pi) / 9, v_max=50.0)  # 80° = 4π/9 radians
        ref_value = 0.3024
    elif geometry_name == 'hemispherical_shell':
        domain_bounds = DomainBounds(u_max=2 * math.pi, v_max=math.pi/2)
    elif geometry_name in ['napkin']:
        domain_bounds = DomainBounds(u_max=1.0, v_max=1.0)
        ref_value = 1.0 # Dummy
    else:
        domain_bounds = DomainBounds(u_max=100.0, v_max=100.0)  # L = 100 m
        ref_value = 0.487
        lr_decay = 5000
    
    rest_surface = RestSurface(geometry_name, test_n_spatial_samples, domain_bounds)
    boundary = Boundary(geometry_name, boundary_condition_name, domain_bounds)
    
    deformation_field = SirenDeformationField(boundary, hidden_features=128, hidden_layers=5).to(device)

    optimizer = torch.optim.Adam(lr=lrate, params=deformation_field.parameters())
    
    surface_geometry = SurfaceGeometry(geometry_name, rest_surface, train_n_spatial_samples)
    material = LinearMaterial(surface_geometry, youngs_modulus, poissons_ratio, thickness)
    
    energy = Energy(surface_geometry, material, geometry_name, domain_bounds, mass_area_density)
    sampler = GridSampler(train_n_spatial_samples, domain_bounds, geometry_name)
    dataloader = DataLoader(sampler, batch_size=1, num_workers=0)
    
    global training_start_time, csv_log_file
    training_start_time = datetime.now()
    csv_log_file = None
    for i in trange(0, n_iterations):
        rest_points = next(iter(dataloader))
        surface_geometry(rest_points)
        deformations = deformation_field(surface_geometry.points)
        mechanical_energy = energy(deformations, i, deformation_field)
        loss = mechanical_energy.mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
        
        if tb_writer:
            tb_writer.add_scalar('scalar/loss', loss, i)
            tb_writer.add_scalar('scalar/mean_deformation', deformations.mean(), i)
        
        new_lrate = lrate * 0.1 ** (i / lr_decay)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate

        if not i % 1000:
            print(f'Iteration: {i}, loss: {loss}, mean_deformation: {deformations.mean()}')
            test(deformation_field, rest_surface, i, geometry_name, ref_value)
            track_loss(i, loss.item(), log_dir, geometry_name)
            track_mean_deformation(i, deformations.mean().item(), log_dir, geometry_name)
        
        # TODO: Add line for saving weights
            
    tb_writer.flush()
    tb_writer.close()

if __name__ == '__main__':
    import argparse

    experiment_configs = {
        'square_plate': {
            'boundary_condition': 'square_plate_clamped',
            'test_n_spatial_samples': 10000,
            'n_iterations': 25001, # 2501
            'youngs_modulus': 10000000,
            'poissons_ratio': 0.0,
            'thickness': 1.0,
            'mass_area_density': None,
            'lrate': 1e-4,
        },
        'napkin': {
            'boundary_condition': 'top_left_fixed',
            'test_n_spatial_samples': 400,
            'n_iterations': 3001,
            'youngs_modulus': 5000,
            'poissons_ratio': 0.25,
            'thickness': 0.0012,
            'mass_area_density': 0.144,
            'lrate': 5e-6,
        },
        'scordelis_lo_roof': {
            'boundary_condition': 'scordelis_lo_roof_boundaries',
            'test_n_spatial_samples': 100,
            'n_iterations': 200001,
            'youngs_modulus': 432000000,
            'poissons_ratio': 0.0,
            'thickness': 0.25,
            'mass_area_density': None,
            'lrate': 1e-4,
        },
        'pinched_cylinder_rigid_diaphragms': {
            'boundary_condition': 'pinched_cylinder_rigid_diaphragms_boundaries',
            'test_n_spatial_samples': 10000,
            'n_iterations': 100001,
            'youngs_modulus': 30,
            'poissons_ratio': 0.3,
            'thickness': 3.0,
            'mass_area_density': None,
            'lrate': 1e-5,
        },
    }

    parser = argparse.ArgumentParser(description="Run TDC experiment")
    parser.add_argument(
        "--experiment",
        type=str,
        choices=experiment_configs.keys(),
        default='pinched_cylinder_rigid_diaphragms',
        help="Name of the experiment to run"
    )
    args = parser.parse_args()

    experiment_name = args.experiment
    config = experiment_configs[experiment_name]
    
    train(
        geometry_name=experiment_name,
        boundary_condition_name=config['boundary_condition'],
        test_n_spatial_samples=config['test_n_spatial_samples'],
        n_iterations=config['n_iterations'],
        youngs_modulus=config['youngs_modulus'],
        poissons_ratio=config['poissons_ratio'],
        thickness=config['thickness'],
        mass_area_density=config['mass_area_density'],
        lrate=config['lrate']
    )
