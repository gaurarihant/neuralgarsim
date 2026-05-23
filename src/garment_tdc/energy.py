import torch
from . import tb
from .differential_operators import jacobian
from .config import device
from .surface_geometry import SurfaceGeometry
from typing import NamedTuple

class Strain(NamedTuple):
    epsilon_M: torch.Tensor
    epsilon_B: torch.Tensor

def tangential_gradient(field: torch.Tensor, ref_geometry: SurfaceGeometry,
                        is_scalar=False):
    if is_scalar:
        field = field.unsqueeze(-1)
    
    grad_u_tilde, status = jacobian(field, ref_geometry.points)

    if status != 0:
        raise ValueError("NaNs in gradient calculation")

    if is_scalar:
        tangential_grad = torch.einsum('bpi,bpij->bpj', grad_u_tilde.squeeze(-2), ref_geometry.projector)  # [B,P,3]
    else:
        tangential_grad = torch.einsum('bpai,bpij->bpaj', grad_u_tilde, ref_geometry.projector)  # [B,P,3,3]

    return tangential_grad

def compute_strain(deformations: torch.Tensor, ref_geometry: SurfaceGeometry, i: int, i_debug: int, nonlinear_strain=True): 

    grad_u = tangential_gradient(deformations, ref_geometry)
    
    epsilon_M_linear = 0.5 * (grad_u + grad_u.transpose(-1, -2))
    
    H = ref_geometry.weingarten
    u_dot_n = torch.einsum('bpi,bpi->bp', deformations, ref_geometry.normals)
    w = torch.einsum('bpij,bpj->bpi', H, deformations) - tangential_gradient(u_dot_n, ref_geometry, is_scalar=True)

    grad_w = tangential_gradient(w, ref_geometry)

    epsilon_B_linear = 0.5 * (grad_w + grad_w.transpose(-1, -2))

    if nonlinear_strain:
        epsilon_M_nonlinear = 0.5 * torch.einsum('bpki,bpkj->bpij', grad_u, grad_u)
        epsilon_B_nonlinear = 0.5 * (torch.einsum('bpki,bpkj->bpij', grad_u, grad_w) + torch.einsum('bpki,bpkj->bpij', grad_w, grad_u))
        
        epsilon_M = epsilon_M_linear + epsilon_M_nonlinear
        epsilon_B = epsilon_B_linear + epsilon_B_nonlinear
    else: 
        epsilon_M = epsilon_M_linear
        epsilon_B = epsilon_B_linear
    
    epsilon_M = torch.einsum('bpik,bpkl,bplj->bpij', ref_geometry.projector, epsilon_M, ref_geometry.projector)
    epsilon_B = torch.einsum('bpik,bpkl,bplj->bpij', ref_geometry.projector, epsilon_B, ref_geometry.projector)
    
    # Output: [B,P,3,3]
    if not i % i_debug and tb.writer:
        print(f"Strain magnitudes - M: {epsilon_M.abs().mean():.6f}, B: {epsilon_B.abs().mean():.6f}")
        print(f"Deformation magnitude: {deformations.abs().mean():.6f}")

    return Strain(epsilon_M, epsilon_B), w
    
class Material():
    def __init__(self, mass_area_density: float, thickness: float, ref_geometry: SurfaceGeometry):
        self.thickness = thickness
        self.mass_area_density = mass_area_density
        self.ref_geometry = ref_geometry
    
    def compute_internal_energy(self, strain: Strain, i: int) -> torch.Tensor:
        raise NotImplementedError
        
class LinearMaterial(Material):
    def __init__(self, args, ref_geometry: SurfaceGeometry):
        super().__init__(args.mass_area_density, args.thickness, ref_geometry)
        self.poissons_ratio = args.poissons_ratio
        self.youngs_modulus = args.youngs_modulus

        self.mu_ = self.youngs_modulus / (2 * (1 + self.poissons_ratio))
        self.lambda_ = (self.youngs_modulus * self.poissons_ratio) / (1 - self.poissons_ratio**2)
    
    def compute_stress(self, strain: torch.Tensor):
        tr_strain = torch.einsum('bpii->bp', strain).unsqueeze(-1).unsqueeze(-1) # [1, 400]
        in_plane_stress = 2 * self.mu_ * strain + self.lambda_ * tr_strain * self.ref_geometry.projector
        # in_plane_stress = torch.einsum('bpik,bpkl,bplj->bpij', projector, in_plane_stress, projector)
        return in_plane_stress

    def compute_internal_energy(self, strain: Strain, i: int) -> torch.Tensor:
        
        sigma_M = self.compute_stress(strain.epsilon_M)
        sigma_B = self.compute_stress(strain.epsilon_B)

        n_tilde = self.thickness * sigma_M
        m_gamma = (self.thickness**3 / 12) * sigma_B

        # Tensor contraction to get energy density
        energy_M = torch.einsum('bpij,bpij->bp', n_tilde, strain.epsilon_M)
        energy_B = torch.einsum('bpij,bpij->bp', m_gamma, strain.epsilon_B)

        if tb.writer:
            tb.writer.add_scalar('loss/membrane_energy', energy_M.mean(), i)
            tb.writer.add_scalar('loss/bending_energy', energy_B.mean(), i)

        return 0.5 * (energy_M + energy_B) # 1

class MaterialOrthotropy(NamedTuple):
    d_1: torch.Tensor
    d_2: torch.Tensor

class NonLinearMaterial(Material):
    def __init__(self, args, ref_geometry: SurfaceGeometry):
        super().__init__(args.mass_area_density, args.thickness, ref_geometry)
        
        self.is_stvk = getattr(args, 'StVK', False) 

        self.a11, self.a12, self.a22, self.G12 = args.a11, args.a12, args.a22, args.G12

        if not self.is_stvk:
            self.d = args.d
            self.mu = [args.mu1, args.mu2, args.mu3, args.mu4]
            self.alpha = [args.alpha1, args.alpha2, args.alpha3, args.alpha4]
            
            self.E_11_min, self.E_11_max, self.E_22_min, self.E_22_max, self.E_12_max = args.E_11_min, args.E_11_max, args.E_22_min, args.E_22_max, args.E_12_max
            self.E_12_min = -self.E_12_max
        self.i_debug = args.i_debug        
    
    def compute_eta(self, j: int, x: torch.Tensor) -> torch.Tensor:
        eta = torch.zeros_like(x)
        for i in range(self.d[j]):
            eta += ((self.mu[j][i] / self.alpha[j][i]) * ((x + 1) ** self.alpha[j][i] - 1))
        return eta
    
    def compute_eta_first_derivative(self, j: int, x: torch.Tensor) -> torch.Tensor:
        eta_first_derivative = torch.zeros_like(x)
        for i in range(self.d[j]):
            eta_first_derivative += (self.mu[j][i] * (x + 1) ** (self.alpha[j][i] - 1))
        return eta_first_derivative
    
    def compute_eta_second_derivative(self, j: int, x: torch.Tensor) -> torch.Tensor: 
        eta_second_derivative = torch.zeros_like(x)
        for i in range(self.d[j]):
            eta_second_derivative += (self.mu[j][i] * (self.alpha[j][i] - 1) * (x + 1) ** (self.alpha[j][i] - 2))
        return eta_second_derivative

    def strain_cutoff_extrapolation(self, E11, E12, E22, E11_c, E12_c, E22_c):
        E11_valid = torch.logical_and(E11 > self.E_11_min, E11 < self.E_11_max)
        E12_valid = torch.logical_and(E12 > self.E_12_min, E12 < self.E_12_max)
        E22_valid = torch.logical_and(E22 > self.E_22_min, E22 < self.E_22_max)
        
        d1_3_E12 = self.compute_eta_first_derivative(3, E12_c ** 2)
        d1_2_E22 = self.compute_eta_first_derivative(2, E22_c ** 2)
        d1_1_Mix = self.compute_eta_first_derivative(1, E11_c * E22_c)
        d1_0_E11 = self.compute_eta_first_derivative(0, E11_c ** 2)
        d2_3_E12 = self.compute_eta_second_derivative(3, E12_c ** 2)
        d2_0_E11 = self.compute_eta_second_derivative(0, E11_c ** 2)
        d2_2_E22 = self.compute_eta_second_derivative(2, E22_c ** 2)
        d2_1_Mix = self.compute_eta_second_derivative(1, E11_c * E22_c)
        
        energy = torch.zeros_like(E11)

        dE12 = (E12 - E12_c)
        shear_term = (2 * self.G12 * E12_c * d1_3_E12 * dE12 + 
                      0.5 * (2 * self.G12 * d1_3_E12 + 4 * self.G12 * E12_c**2 * d2_3_E12) * dE12**2)
        energy += ~E12_valid * shear_term

        dE11 = (E11 - E11_c)
        term1 = (self.a11 * E11_c * d1_0_E11 + self.a12 * E22_c * d1_1_Mix) * dE11 + \
                0.5 * (self.a11 * d1_0_E11 + 2 * self.a11 * E11_c**2 * d2_0_E11 + self.a12 * E22_c**2 * d2_1_Mix) * dE11**2
        energy += ~E11_valid * term1

        dE22 = (E22 - E22_c)
        term2 = (self.a22 * E22_c * d1_2_E22 + self.a12 * E11_c * d1_1_Mix) * dE22 + \
                0.5 * (self.a22 * d1_2_E22 + 2 * self.a22 * E22_c**2 * d2_2_E22 + self.a12 * E11_c**2 * d2_1_Mix) * dE22**2
        energy += ~E22_valid * term2

        energy += (~E11_valid * ~E22_valid) * (self.a12 * d1_1_Mix + self.a12 * E11_c * E22_c * d2_1_Mix) * dE11 * dE22
        return energy

    def compute_internal_energy(self, strain: Strain, material_directions, i: int, xi__3: float) -> torch.Tensor:

        E_shell_basis = strain.epsilon_M + xi__3 * strain.epsilon_B
        
        d1 = material_directions.d_1
        d2 = material_directions.d_2

        Td1 = torch.einsum('bnij,bnj->bni', E_shell_basis, d1)
        Td2 = torch.einsum('bnij,bnj->bni', E_shell_basis, d2)

        E11 = torch.einsum('bni,bni->bn', d1, Td1)
        E22 = torch.einsum('bni,bni->bn', d2, Td2)
        E12 = torch.einsum('bni,bni->bn', d1, Td2)

        if self.is_stvk:
            psi = (0.5 * self.a11 * E11**2 + 
                   0.5 * self.a22 * E22**2 + 
                   2.0 * 0.5 * self.G12 * E12**2 + 
                   self.a12 * E11 * E22)
        else:
            E11_c = torch.clamp(E11, self.E_11_min, self.E_11_max)
            E12_c = torch.clamp(E12, self.E_12_min, self.E_12_max)
            E22_c = torch.clamp(E22, self.E_22_min, self.E_22_max)

            psi = (self.a11 * 0.5 * self.compute_eta(0, E11_c**2) + 
                   self.a12 * self.compute_eta(1, E11_c * E22_c) + 
                   self.a22 * 0.5 * self.compute_eta(2, E22_c**2) + 
                   self.G12 * self.compute_eta(3, E12_c**2))
            
            psi += self.strain_cutoff_extrapolation(E11, E12, E22, E11_c, E12_c, E22_c)

        return psi


class ShellEnergy:
    def __init__(
        self,
        ref_geometry: SurfaceGeometry, 
        material: Material, 
        uniform_force: list,
        i_debug: int
    ):
        self.ref_geometry = ref_geometry
        self.material = material
        self.i_debug = i_debug
        
        self.uniform_force = uniform_force

        self.external_load = None
        self.rest_positions = None
        self._forces_initialized = False
    
    def _initialize_forces(self):
        if self._forces_initialized:
            return
        
        self.rest_positions = self.ref_geometry.points.detach()
        self.external_load = torch.zeros(
            1, self.ref_geometry.n_spatial_samples, 3, 
            device=device
        )

        self._apply_uniform_force(self.uniform_force)
        
        self.external_load *= self.material.mass_area_density
        self._forces_initialized = True

    def _apply_uniform_force(self, uniform_force: list):
        force_vec = torch.tensor(uniform_force, device=device).view(1, 1, 3)
        n_verts = self.ref_geometry.n_spatial_samples
        self.external_load += force_vec.expand(1, n_verts, 3)

    def __call__(self, deformations: torch.Tensor, i: int) -> torch.Tensor:   
        if not self._forces_initialized:
            self._initialize_forces()

        strain, w = compute_strain(deformations, self.ref_geometry, i, self.i_debug)
        
        area_element = self.ref_geometry.area_element.unsqueeze(0)
        
        if isinstance(self.material, LinearMaterial):        
            hyperelastic_strain_energy_mid = self.material.compute_internal_energy(strain, i)
            external_energy_mid = torch.einsum('ijk,ijk->ij', self.external_load, deformations)
            mechanical_energy = (hyperelastic_strain_energy_mid - external_energy_mid) * area_element
            return mechanical_energy
            
        elif isinstance(self.material, NonLinearMaterial):
            d_1 = self.ref_geometry.a_1
            d_2 = torch.linalg.cross(self.ref_geometry.normals, d_1)
            
            material_directions = MaterialOrthotropy(d_1, d_2)

            h = self.material.thickness
            xi_top = -0.5 * h
            xi_mid = 0.0
            xi_bot = 0.5 * h
            
            psi_top = self.material.compute_internal_energy(strain, material_directions, i, xi_top)
            ext_top = torch.einsum('ijk,ijk->ij', self.external_load, deformations + xi_top * w)

            psi_mid = self.material.compute_internal_energy(strain, material_directions, i, xi_mid)
            ext_mid = torch.einsum('ijk,ijk->ij', self.external_load, deformations + xi_mid * w)

            psi_bot = self.material.compute_internal_energy(strain, material_directions, i, xi_bot)
            ext_bot = torch.einsum('ijk,ijk->ij', self.external_load, deformations + xi_bot * w)
            
            total_top = (psi_top - ext_top)
            total_mid = (psi_mid - ext_mid)
            total_bot = (psi_bot - ext_bot)
            
            mechanical_energy = (h / 6.0) * (total_top + 4.0 * total_mid + total_bot) * area_element
                
            return mechanical_energy

        return mechanical_energy
