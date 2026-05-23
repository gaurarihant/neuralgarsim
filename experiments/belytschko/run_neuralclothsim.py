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


class CurvilinearSpace(NamedTuple):
    xi__1_max: float
    xi__2_max: float
    
class GridSampler(Dataset):
    def __init__(self, n_spatial_samples: int, curvilinear_space: CurvilinearSpace, geometry_name: str):
        self.n_spatial_samples = n_spatial_samples
        self.curvilinear_space = curvilinear_space
        self.geometry_name = geometry_name
        
        self.spatial_sidelen = math.isqrt(n_spatial_samples)
        self.cell_curvilinear_coords = get_mgrid((self.spatial_sidelen, self.spatial_sidelen), stratified=True, dim=2)            

    def __len__(self):
        return 1
    
    def __getitem__(self, idx):    
        if idx > 0: raise IndexError
         
        curvilinear_coords = self.cell_curvilinear_coords.clone()
        
        t_rand_spatial = torch.rand([curvilinear_coords.shape[0], 2], device=device) / self.spatial_sidelen        
        curvilinear_coords += t_rand_spatial
        
        curvilinear_coords[...,0] *= self.curvilinear_space.xi__1_max
        curvilinear_coords[...,1] *= self.curvilinear_space.xi__2_max
        curvilinear_coords.requires_grad_(True)

        return curvilinear_coords
    
class Boundary:
    def __init__(self, geometry_name: str, boundary_condition_name: str, curvilinear_space: CurvilinearSpace):
        self.geometry_name = geometry_name
        self.boundary_condition_name = boundary_condition_name
        self.curvilinear_space = curvilinear_space
        self.boundary_support = 0.01

    def periodic_condition_and_normalization(self, curvilinear_coords: torch.Tensor) -> torch.Tensor:
        if self.geometry_name in ['pinched_cylinder_rigid_diaphragms', 'compressed_cylinder', 'twisted_cylinder']:
            normalized_coords = torch.cat([
                (torch.cos(curvilinear_coords[...,0:1]) + 1)/2, 
                (torch.sin(curvilinear_coords[...,0:1]) + 1)/2, 
                curvilinear_coords[...,1:2]/self.curvilinear_space.xi__2_max
            ], dim=2)
        else:
            normalized_coords = torch.cat([
                curvilinear_coords[...,0:1]/self.curvilinear_space.xi__1_max, 
                curvilinear_coords[...,1:2]/self.curvilinear_space.xi__2_max
            ], dim=2)
        return normalized_coords
    
    def dirichlet_condition(self, deformations: torch.Tensor, curvilinear_coords: torch.Tensor) -> torch.Tensor:        
        match self.boundary_condition_name:
            case 'top_left_fixed':
                top_left_corner = torch.exp(-(curvilinear_coords[...,0:1] ** 2 + (curvilinear_coords[...,1:2] - self.curvilinear_space.xi__2_max) ** 2)/self.boundary_support)
                deformations = deformations * (1 - top_left_corner)
            case 'square_plate_clamped':
                L1 = self.curvilinear_space.xi__1_max
                L2 = self.curvilinear_space.xi__2_max
                left_edge = curvilinear_coords[...,0:1] / L1
                right_edge = curvilinear_coords[...,1:2] / L2
                edge_mask = left_edge * right_edge * (1 - left_edge) * (1 - right_edge)
                deformations = deformations * edge_mask
            case 'scordelis_lo_roof_boundaries':
                L2 = self.curvilinear_space.xi__2_max
                bottom_end = curvilinear_coords[..., 1:2] / L2
                top_end = (self.curvilinear_space.xi__2_max - curvilinear_coords[..., 1:2]) / L2
                end_mask = bottom_end * top_end
    
                deformations = torch.cat([
                    deformations[...,0:1] * end_mask,
                    deformations[...,1:2] * end_mask,
                    deformations[...,2:3]
                ], dim=2)
            case 'pinched_cylinder_rigid_diaphragms_boundaries':
                # For rigid diaphragm: all displacements constrained at ends
                L2 = self.curvilinear_space.xi__2_max
                bottom_end = curvilinear_coords[..., 1:2] / L2
                top_end = (self.curvilinear_space.xi__2_max - curvilinear_coords[..., 1:2]) / L2
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
   
class Siren(nn.Module):
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
    
    def forward(self, curvilinear_coords: torch.Tensor) -> torch.Tensor:                

        normalized_coords = self.boundary.periodic_condition_and_normalization(curvilinear_coords)        
        deformations = self.net(normalized_coords)
        deformations = self.boundary.dirichlet_condition(deformations, curvilinear_coords)

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
    
class ReferenceMidSurface():
    def __init__(self, geometry_name: str, test_n_spatial_samples: int, curvilinear_space: CurvilinearSpace):
        self.geometry_name = geometry_name
        
        test_spatial_sidelen = math.isqrt(test_n_spatial_samples)
        curvilinear_coords = get_mgrid((test_spatial_sidelen, test_spatial_sidelen), stratified=False, dim=2)[None]
            
        curvilinear_coords[...,0] *= curvilinear_space.xi__1_max
        curvilinear_coords[...,1] *= curvilinear_space.xi__2_max
        
        vertices = self(curvilinear_coords)[0]
        faces = torch.tensor(generate_mesh_topology(test_spatial_sidelen), device=device)
        texture = TexturesUV(maps=torch.empty(1, 1, 1, 1, device=device), faces_uvs=[faces], verts_uvs=curvilinear_coords)
        self.template_mesh = Meshes(verts=[vertices], faces=[faces], textures=texture).to(device)
        
    def __call__(self, curvilinear_coords: torch.Tensor) -> torch.Tensor:
        xi__1 = curvilinear_coords[...,0] 
        xi__2 = curvilinear_coords[...,1]
        
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
        plt.xlabel('Curvilinear Coord 1')
        plt.ylabel('Curvilinear Coord 2')

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

class ReferenceGeometry(): 
    def __init__(self, reference_midsurface: ReferenceMidSurface, n_spatial_samples: int):
        self.n_spatial_samples = n_spatial_samples 
        self.reference_midsurface = reference_midsurface

    def __call__(self, curvilinear_coords: torch.Tensor):
        self.curvilinear_coords = curvilinear_coords
        self.midsurface_positions = self.reference_midsurface(self.curvilinear_coords)
        self.base_vectors()
        self.metric_tensor()
        self.curvature_tensor()
        self.christoffel_symbol()
        self.coord_transform()

    def base_vectors(self):
        base_vectors = jacobian(self.midsurface_positions, self.curvilinear_coords)[0]
        self.a_1 = base_vectors[...,0]
        self.a_2 = base_vectors[...,1]
        self.a_3 = normalize(torch.linalg.cross(self.a_1, self.a_2), dim=2)
              
    def metric_tensor(self):
        self.a_1_1 = torch.einsum('ijk,ijk->ij', self.a_1, self.a_1)
        self.a_1_2 = torch.einsum('ijk,ijk->ij', self.a_1, self.a_2)
        self.a_2_2 = torch.einsum('ijk,ijk->ij', self.a_2, self.a_2)

        self.a = self.a_1_1 * self.a_2_2 - (self.a_1_2 ** 2)
        self.a__1__1 = self.a_2_2 / self.a
        self.a__2__2 = self.a_1_1 / self.a
        self.a__1__2 = -1 * self.a_1_2 / self.a
        self.a__2__1 = self.a__1__2
        self.a__1 = torch.einsum('ij,ijk->ijk', self.a__1__1, self.a_1) + torch.einsum('ij,ijk->ijk', self.a__1__2, self.a_2)
        self.a__2 = torch.einsum('ij,ijk->ijk', self.a__1__2, self.a_1) + torch.einsum('ij,ijk->ijk', self.a__2__2, self.a_2)        
    
    def curvature_tensor(self):        
        self.a_3pd = jacobian(self.a_3, self.curvilinear_coords)[0]
        self.b_1_1 = -1 * torch.einsum('ijk,ijk->ij', self.a_1, self.a_3pd[...,0])
        self.b_1_2 = -1 * torch.einsum('ijk,ijk->ij', self.a_1, self.a_3pd[...,1])
        self.b_2_2 = -1 * torch.einsum('ijk,ijk->ij', self.a_2, self.a_3pd[...,1])
        self.b_2_1 = self.b_1_2

        self.b_1__1 = self.b_1_1 * self.a__1__1 + self.b_1_2 * self.a__2__1
        self.b_1__2 = self.b_1_1 * self.a__1__2 + self.b_1_2 * self.a__2__2
        self.b_2__1 = self.b_2_1 * self.a__1__1 + self.b_2_2 * self.a__2__1
        self.b_2__2 = self.b_2_1 * self.a__1__2 + self.b_2_2 * self.a__2__2

    def coord_transform(self):
        with torch.no_grad():
            contravariant_coord_2_cartesian = torch.stack([self.a_1, self.a_2, self.a_3], dim=3)
            self.cartesian_coord_2_contravariant = torch.linalg.inv(contravariant_coord_2_cartesian)
            
        covariant_coord_2_cartesian = torch.stack([self.a__1, self.a__2, self.a_3], dim=3)
        self.cartesian_coord_2_covariant = torch.linalg.inv(covariant_coord_2_cartesian)    
    
    def christoffel_symbol(self):
        a_1pd = jacobian(self.a_1, self.curvilinear_coords)[0]
        a_2pd = jacobian(self.a_2, self.curvilinear_coords)[0]

        with torch.no_grad():
            self.gamma_11__1 = torch.einsum('ijk,ijk->ij', a_1pd[...,0], self.a__1)
            self.gamma_12__1 = torch.einsum('ijk,ijk->ij', a_1pd[...,1], self.a__1)
            self.gamma_22__1 = torch.einsum('ijk,ijk->ij', a_2pd[...,1], self.a__1)

            self.gamma_11__2 = torch.einsum('ijk,ijk->ij', a_1pd[...,0], self.a__2)
            self.gamma_12__2 = torch.einsum('ijk,ijk->ij', a_1pd[...,1], self.a__2)
            self.gamma_22__2 = torch.einsum('ijk,ijk->ij', a_2pd[...,1], self.a__2)
            self.gamma_21__1 = self.gamma_12__1
            self.gamma_21__2 = self.gamma_12__2

    @torch.no_grad()
    def elastic_tensor(self, poissons_ratio: float):
        H__1111 = poissons_ratio * self.a__1__1 * self.a__1__1 + 0.5 * (1 - poissons_ratio) * (self.a__1__1 * self.a__1__1 + self.a__1__1 * self.a__1__1)
        H__1112 = poissons_ratio * self.a__1__1 * self.a__1__2 + 0.5 * (1 - poissons_ratio) * (self.a__1__1 * self.a__1__2 + self.a__1__2 * self.a__1__1)
        H__1122 = poissons_ratio * self.a__1__1 * self.a__2__2 + 0.5 * (1 - poissons_ratio) * (self.a__1__2 * self.a__1__2 + self.a__1__2 * self.a__1__2)
        H__1212 = poissons_ratio * self.a__1__2 * self.a__1__2 + 0.5 * (1 - poissons_ratio) * (self.a__1__1 * self.a__2__2 + self.a__1__2 * self.a__2__1)
        H__1222 = poissons_ratio * self.a__1__2 * self.a__2__2 + 0.5 * (1 - poissons_ratio) * (self.a__1__2 * self.a__2__2 + self.a__1__2 * self.a__2__2)
        H__2222 = poissons_ratio * self.a__2__2 * self.a__2__2 + 0.5 * (1 - poissons_ratio) * (self.a__2__2 * self.a__2__2 + self.a__2__2 * self.a__2__2)
        return H__1111, H__1112, H__1122, H__1212, H__1222, H__2222
        
    def shell_base_vectors(self, xi__3: float):
        g_1_1 = self.a_1_1 - 2 * xi__3 * self.b_1_1
        g_1_2 = self.a_1_2 - 2 * xi__3 * self.b_1_2
        g_2_2 = self.a_2_2 - 2 * xi__3 * self.b_2_2
        
        g_1 = self.a_1 + xi__3 * self.a_3pd[...,0]
        g_2 = self.a_2 + xi__3 * self.a_3pd[...,1]
        g_covariant_matrix = torch.stack([torch.stack([g_1_1, g_1_2], dim=2), torch.stack([g_1_2, g_2_2], dim=2)], dim=2) 
        g_contravariant_matrix = torch.linalg.inv(g_covariant_matrix)
        g__1 = torch.einsum('ij,ijk->ijk', g_contravariant_matrix[...,0,0], g_1) + torch.einsum('ij,ijk->ijk', g_contravariant_matrix[...,0,1], g_2)
        g__2 = torch.einsum('ij,ijk->ijk', g_contravariant_matrix[...,1,0], g_1) + torch.einsum('ij,ijk->ijk', g_contravariant_matrix[...,1,1], g_2)

        g__1__1 = torch.einsum('ijk,ijl->ijkl', g__1, g__1)
        g__1__2 = torch.einsum('ijk,ijl->ijkl', g__1, g__2)
        g__2__1 = torch.einsum('ijk,ijl->ijkl', g__2, g__1)
        g__2__2 = torch.einsum('ijk,ijl->ijkl', g__2, g__2)
        return g__1__1, g__1__2, g__2__1, g__2__2  
    
class Strain(NamedTuple):
    epsilon_1_1: torch.Tensor
    epsilon_1_2: torch.Tensor
    epsilon_2_2: torch.Tensor
    kappa_1_1: torch.Tensor
    kappa_1_2: torch.Tensor
    kappa_2_2: torch.Tensor
    
def covariant_first_derivative_of_covariant_first_order_tensor(covariant_vectors, ref_geometry):
    vpd = jacobian(torch.stack(covariant_vectors, dim=2), ref_geometry.curvilinear_coords)[0]
    v_1, v_2 = covariant_vectors   
    
    v1cd1 = vpd[...,0,0] - v_1 * ref_geometry.gamma_11__1 - v_2 * ref_geometry.gamma_11__2
    v1cd2 = vpd[...,0,1] - v_1 * ref_geometry.gamma_12__1 - v_2 * ref_geometry.gamma_12__2
    v2cd1 = vpd[...,1,0] - v_1 * ref_geometry.gamma_21__1 - v_2 * ref_geometry.gamma_21__2
    v2cd2 = vpd[...,1,1] - v_1 * ref_geometry.gamma_22__1 - v_2 * ref_geometry.gamma_22__2
    
    return [v1cd1, v1cd2, v2cd1, v2cd2]

def covariant_first_derivative_of_covariant_second_order_tensor(covariant_matrix, ref_geometry):
    phipd = jacobian(torch.stack(covariant_matrix, dim=2), ref_geometry.curvilinear_coords)[0] 
    phi_1_1, phi_1_2, phi_2_1, phi_2_2 = covariant_matrix
    
    phi_1_1cd1 = phipd[...,0,0] - phi_1_1 * ref_geometry.gamma_11__1 - phi_2_1 * ref_geometry.gamma_11__2 - phi_1_1 * ref_geometry.gamma_11__1 - phi_1_2 * ref_geometry.gamma_11__2
    phi_1_1cd2 = phipd[...,0,1] - phi_1_1 * ref_geometry.gamma_12__1 - phi_2_1 * ref_geometry.gamma_12__2 - phi_1_1 * ref_geometry.gamma_12__1 - phi_1_2 * ref_geometry.gamma_12__2
    
    phi_1_2cd1 = phipd[...,1,0] - phi_1_2 * ref_geometry.gamma_11__1 - phi_2_2 * ref_geometry.gamma_11__2 - phi_1_1 * ref_geometry.gamma_21__1 - phi_1_2 * ref_geometry.gamma_21__2
    phi_1_2cd2 = phipd[...,1,1] - phi_1_2 * ref_geometry.gamma_12__1 - phi_2_2 * ref_geometry.gamma_12__2 - phi_1_1 * ref_geometry.gamma_22__1 - phi_1_2 * ref_geometry.gamma_22__2
    
    phi_2_1cd1 = phipd[...,2,0] - phi_1_1 * ref_geometry.gamma_21__1 - phi_2_1 * ref_geometry.gamma_21__2 - phi_2_1 * ref_geometry.gamma_11__1 - phi_2_2 * ref_geometry.gamma_11__2
    phi_2_1cd2 = phipd[...,2,1] - phi_1_1 * ref_geometry.gamma_22__1 - phi_2_1 * ref_geometry.gamma_22__2 - phi_2_1 * ref_geometry.gamma_12__1 - phi_2_2 * ref_geometry.gamma_12__2
    
    phi_2_2cd1 = phipd[...,3,0] - phi_1_2 * ref_geometry.gamma_21__1 - phi_2_2 * ref_geometry.gamma_21__2 - phi_2_1 * ref_geometry.gamma_21__1 - phi_2_2 * ref_geometry.gamma_21__2
    phi_2_2cd2 = phipd[...,3,1] - phi_1_2 * ref_geometry.gamma_22__1 - phi_2_2 * ref_geometry.gamma_22__2 - phi_2_1 * ref_geometry.gamma_22__1 - phi_2_2 * ref_geometry.gamma_22__2
    
    return phi_1_1cd1, phi_1_1cd2, phi_1_2cd1, phi_1_2cd2, phi_2_1cd1, phi_2_1cd2, phi_2_2cd1, phi_2_2cd2

def compute_strain(deformations: torch.Tensor, ref_geometry: ReferenceGeometry, i: int, nonlinear_strain=True): 
    
    deformations_local = torch.einsum('ijkl,ijl->ijk', ref_geometry.cartesian_coord_2_covariant, deformations)
    u_1, u_2, u_3 = deformations_local[...,0], deformations_local[...,1], deformations_local[...,2]

    u1cd1, u1cd2, u2cd1, u2cd2 = covariant_first_derivative_of_covariant_first_order_tensor([u_1, u_2], ref_geometry) 
    
    phi_1_1 = u1cd1 - ref_geometry.b_1_1 * u_3
    phi_1_2 = u2cd1 - ref_geometry.b_2_1 * u_3
    phi_2_1 = u1cd2 - ref_geometry.b_1_2 * u_3
    phi_2_2 = u2cd2 - ref_geometry.b_2_2 * u_3

    u_3pd = jacobian(u_3[...,None], ref_geometry.curvilinear_coords)[0]
    phi_1_3 = u_3pd[...,0,0] + ref_geometry.b_1__1 * u_1 + ref_geometry.b_1__2 * u_2
    phi_2_3 = u_3pd[...,0,1] + ref_geometry.b_2__1 * u_1 + ref_geometry.b_2__2 * u_2
    
    epsilon_1_1_linear = phi_1_1
    epsilon_1_2_linear = 0.5 * (phi_1_2 + phi_2_1)
    epsilon_2_2_linear = phi_2_2
    
    phi_1_3cd1, phi_1_3cd2, phi_2_3cd1, phi_2_3cd2 = covariant_first_derivative_of_covariant_first_order_tensor([phi_1_3, phi_2_3], ref_geometry)
    
    kappa_1_1_linear = -1 * phi_1_3cd1 - ref_geometry.b_1__1 * phi_1_1 - ref_geometry.b_1__2 * phi_1_2 
    kappa_1_2_linear = -1 * phi_1_3cd2 - ref_geometry.b_2__1 * phi_1_1 - ref_geometry.b_2__2 * phi_1_2 
    kappa_2_2_linear = -1 * phi_2_3cd2 - ref_geometry.b_2__1 * phi_2_1 - ref_geometry.b_2__2 * phi_2_2 
    
    epsilon_1_1, epsilon_1_2, epsilon_2_2, kappa_1_1, kappa_1_2, kappa_2_2 = epsilon_1_1_linear, epsilon_1_2_linear, epsilon_2_2_linear, kappa_1_1_linear, kappa_1_2_linear, kappa_2_2_linear
    
    return Strain(epsilon_1_1, epsilon_1_2, epsilon_2_2, kappa_1_1, kappa_1_2, kappa_2_2)
    
class Material():
    def __init__(self, thickness: float, ref_geometry: ReferenceGeometry):
        self.thickness = thickness
        self.ref_geometry = ref_geometry
    
    def compute_internal_energy(self, strain: Strain) -> torch.Tensor:
        raise NotImplementedError
        
class LinearMaterial(Material):
    def __init__(self, ref_geometry: ReferenceGeometry, youngs_modulus: float, poissons_ratio: float, thickness: float):
        super().__init__(thickness, ref_geometry)
        self.youngs_modulus = youngs_modulus
        self.thickness = thickness
        self.poissons_ratio = poissons_ratio
        self.D = (self.youngs_modulus * self.thickness) / (1 - self.poissons_ratio ** 2)
        self.B = (self.thickness ** 2) * self.D / 12 
    
    def compute_internal_energy(self, strain: Strain) -> torch.Tensor:
        H__1111, H__1112, H__1122, H__1212, H__1222, H__2222 = self.ref_geometry.elastic_tensor(self.poissons_ratio)
            
        n__1__1 = H__1111 * strain.epsilon_1_1 + 2 * H__1112 * strain.epsilon_1_2 + H__1122 * strain.epsilon_2_2
        n__1__2 = H__1112 * strain.epsilon_1_1 + 2 * H__1212 * strain.epsilon_1_2 + H__1222 * strain.epsilon_2_2
        n__2__1 = n__1__2
        n__2__2 = H__1122 * strain.epsilon_1_1 + 2 * H__1222 * strain.epsilon_1_2 + H__2222 * strain.epsilon_2_2

        m__1__1 = H__1111 * strain.kappa_1_1 + 2 * H__1112 * strain.kappa_1_2 + H__1122 * strain.kappa_2_2
        m__1__2 = H__1112 * strain.kappa_1_1 + 2 * H__1212 * strain.kappa_1_2 + H__1222 * strain.kappa_2_2
        m__2__1 = m__1__2
        m__2__2 = H__1122 * strain.kappa_1_1 + 2 * H__1222 * strain.kappa_1_2 + H__2222 * strain.kappa_2_2
        
        hyperelastic_strain_energy = 0.5 * (self.D * (strain.epsilon_1_1 * n__1__1 + strain.epsilon_1_2 * n__1__2 + strain.epsilon_1_2 * n__2__1 + strain.epsilon_2_2 * n__2__2) + self.B * (strain.kappa_1_1 * m__1__1 + strain.kappa_1_2 * m__1__2 + strain.kappa_1_2 * m__2__1 + strain.kappa_2_2 * m__2__2))
        
        return hyperelastic_strain_energy

class Energy:
    def __init__(self, ref_geometry: ReferenceGeometry, material: Material, geometry_name: str, curvilinear_space: CurvilinearSpace, mass_area_density):
        self.ref_geometry = ref_geometry
        self.material = material
        self.geometry_name = geometry_name
        self.curvilinear_space = curvilinear_space
        self.get_distributed_load(mass_area_density)
        self.get_point_loads()

    def get_point_loads(self):
        match self.geometry_name:
            case 'pinched_cylinder_rigid_diaphragms':
                coords = torch.tensor([[[math.pi/2, 300.0], 
                                        [3*math.pi/2, 300.0]]], device=device)
                loads = torch.tensor([[[0., 0., -1.], [0., 0., 1.]]], device=device)
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

    def __call__(self, deformations: torch.Tensor, i: int, ndf: Siren = None) -> torch.Tensor:   
        strain = compute_strain(deformations, self.ref_geometry, i)
        
        hyperelastic_strain_energy = self.material.compute_internal_energy(strain)
        
        # Handle point loads for pinched cylinder
        if self.geometry_name in ['pinched_cylinder_rigid_diaphragms']: 
            displacements_load_point = ndf(self.point_coords)

            ext_energy = -torch.einsum('ijk,ijk->ij', self.point_loads, displacements_load_point)
            
            mechanical_energy = 2 * math.pi * 300 * 600 * hyperelastic_strain_energy.mean() + ext_energy.sum()
            # mechanical_energy = (hyperelastic_strain_energy.mean() * torch.sqrt(self.ref_geometry.a)) - ext_energy.sum()
        else:
            # Distributed loads
            external_energy_mid = torch.einsum('ijk,ijk->ij', self.external_load, deformations)
            mechanical_energy = (hyperelastic_strain_energy - external_energy_mid)
            mechanical_energy = mechanical_energy * torch.sqrt(self.ref_geometry.a)
        
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

def test(ndf: Siren, reference_midsurface: ReferenceMidSurface, i: int, geometry_name: str, ref_value: float):
    test_deformations = ndf(reference_midsurface.template_mesh.textures.verts_uvs_padded())
    test_deformed_positions = reference_midsurface.template_mesh.verts_padded() + test_deformations
    
    tb_writer.add_mesh('simulated_states', test_deformed_positions, faces=reference_midsurface.template_mesh.textures.faces_uvs_padded(), global_step=i)
    
    if i % 1000 == 0:
        test_meshes_dir = os.path.join('logs', geometry_name, 'test_meshes')
        plots_dir = os.path.join('logs', geometry_name, 'plots')
        tracks_dir = os.path.join('logs', geometry_name)
        
        # save_meshes(test_deformed_positions, reference_midsurface.template_mesh.textures.faces_uvs_padded()[0], 
        #            [0], test_meshes_dir, f'mesh_{i}', test_deformed_positions.shape[1])
        
        if geometry_name in ['square_plate', 'napkin']:
            plot_displacement(reference_midsurface.template_mesh.textures.verts_uvs_padded(), test_deformations, 2, os.path.join(plots_dir, 'displacement_plots'), i, 
                            math.isqrt(test_deformed_positions.shape[1]), geometry_name)
        elif geometry_name == 'scordelis_lo_roof':
            plot_displacement(reference_midsurface.template_mesh.textures.verts_uvs_padded(), test_deformations, 1, os.path.join(plots_dir, 'displacement_plots'), i, 
                            math.isqrt(test_deformed_positions.shape[1]), geometry_name)
        elif geometry_name in ['pinched_cylinder_rigid_diaphragms']:
            plot_displacement(reference_midsurface.template_mesh.textures.verts_uvs_padded(), test_deformations, 2, os.path.join(plots_dir, 'displacement_plots'), i, 
                            math.isqrt(test_deformed_positions.shape[1]), geometry_name)
        
        # For pinched cylinder cases, check displacement at load point
        if geometry_name in ['pinched_cylinder_rigid_diaphragms']:
            load_coords = torch.tensor([[[math.pi/2, 300.0]]], device=device, requires_grad=True)
            load_point_disp = ndf(load_coords)
            max_disp = torch.abs(load_point_disp[0,0,2]).item() * 1e-5
        else:
            max_disp = torch.max(torch.abs(test_deformations)).item()
            
        tb_writer.add_scalar('scalar/max_displacement', max_disp, i)
        print(f'Max displacement at step {i}: {max_disp}')
        track_displacement(i, max_disp, tracks_dir, geometry_name, reference_value=ref_value) 

def train(geometry_name, boundary_condition_name, test_n_spatial_samples, n_iterations, 
          youngs_modulus=5000, poissons_ratio=0.25, thickness=0.0012, mass_area_density=0.144, lrate=5e-6):
    
    log_dir = os.path.join('logs', geometry_name)
    global tb_writer
    tb_writer = SummaryWriter(log_dir)
    
    # Create directories for visualizations
    test_meshes_dir = os.path.join(log_dir, 'test_meshes')
    plots_dir = os.path.join(log_dir, 'plots')
    os.makedirs(test_meshes_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    lr_decay = 50000
    train_n_spatial_samples = 100
    # Set curvilinear space based on geometry
    if geometry_name in ['pinched_cylinder_rigid_diaphragms', 'twisted_cylinder']:
        lr_decay = 20000
        train_n_spatial_samples = 10000
        curvilinear_space = CurvilinearSpace(xi__1_max=2 * math.pi, xi__2_max=600.0)  # L = 600 for cylinder
        if geometry_name == 'pinched_cylinder_rigid_diaphragms':
            ref_value = 1.825e-5
    elif geometry_name == 'scordelis_lo_roof':
        curvilinear_space = CurvilinearSpace(xi__1_max=(4 * math.pi) / 9, xi__2_max=50.0)  # 80° = 4π/9 radians
        ref_value = 0.3024
    elif geometry_name == 'hemispherical_shell':
        curvilinear_space = CurvilinearSpace(xi__1_max=2 * math.pi, xi__2_max=math.pi/2)
    elif geometry_name in ['napkin']:
        curvilinear_space = CurvilinearSpace(xi__1_max=1.0, xi__2_max=1.0)
        ref_value = 1.0 # Dummy
    else:
        curvilinear_space = CurvilinearSpace(xi__1_max=100.0, xi__2_max=100.0)  # L = 100 m
        ref_value = 0.487
        lr_decay = 5000
    
    reference_midsurface = ReferenceMidSurface(geometry_name, test_n_spatial_samples, curvilinear_space)
    boundary = Boundary(geometry_name, boundary_condition_name, curvilinear_space)
    
    if geometry_name in ['pinched_cylinder_rigid_diaphragms', 'pinched_cylinder', 'compressed_cylinder', 'twisted_cylinder']:
        in_features = 3 
    else:
        in_features = 2
    
    ndf = Siren(boundary, in_features=in_features, hidden_features=128, hidden_layers=5).to(device)

    optimizer = torch.optim.Adam(lr=lrate, params=ndf.parameters())
    
    reference_geometry = ReferenceGeometry(reference_midsurface, train_n_spatial_samples)
    material = LinearMaterial(reference_geometry, youngs_modulus, poissons_ratio, thickness)
    
    energy = Energy(reference_geometry, material, geometry_name, curvilinear_space, mass_area_density)
    sampler = GridSampler(train_n_spatial_samples, curvilinear_space, geometry_name)
    dataloader = DataLoader(sampler, batch_size=1, num_workers=0)
        
    global training_start_time, csv_log_file
    training_start_time = datetime.now()
    csv_log_file = None
    for i in trange(0, n_iterations):
        curvilinear_coords = next(iter(dataloader))
        reference_geometry(curvilinear_coords)
        deformations = ndf(reference_geometry.curvilinear_coords)
        mechanical_energy = energy(deformations, i, ndf)
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
            test(ndf, reference_midsurface, i, geometry_name, ref_value)
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
