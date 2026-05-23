import os
import math
import torch
import torch.nn.functional as F
import numpy as np
import logging
from tqdm import trange
from pytorch3d.io import save_obj, load_obj
from pytorch3d.ops import sample_points_from_meshes
from .differential_operators import gradient, jacobian, hessian
import open3d as o3d
import open3d.core as o3c

from . import tb
from .config import device
from .models import SirenGarment
from .sampling import get_mgrid, DomainBounds
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.renderer.mesh.textures import TexturesUV, TexturesVertex
from pytorch3d.transforms import matrix_to_quaternion
from plyfile import PlyData, PlyElement

def o3c_to_torch(tensor: o3c.Tensor) -> torch.Tensor:
    return torch.utils.dlpack.from_dlpack(tensor.to_dlpack())

def torch_to_o3c(tensor: torch.Tensor) -> o3c.Tensor:
    return o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(tensor))

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

class RestSurface():
    def __init__(self, args, domain_bounds: DomainBounds | None, log: logging.Logger):
        self.reference_geometry_name = args.reference_geometry_name
        self.boundary_rest_points = None
        self.reference_mlp = None
        self.log = log
        self.rep_type = args.rep_type
        if self.reference_geometry_name == 'mesh':
            vertices, faces, aux = load_obj(args.reference_geometry_source, load_textures=False)
    
            if aux.verts_uvs is not None and len(aux.verts_uvs) > 0:
                if aux.texture_images and len(aux.texture_images) > 0:
                    map_image = list(aux.texture_images.values())[0]
                else:
                    map_image = torch.ones((1, 1, 3), device=vertices.device)

                texture = TexturesUV(
                    maps=[map_image], 
                    faces_uvs=[faces.textures_idx], 
                    verts_uvs=[aux.verts_uvs]
                )
            else:
                texture = TexturesVertex(verts_features=vertices.unsqueeze(0))          

            if args.reference_geometry_type == 'gaussians':
                self.log.info("Generating 3D Gaussians from mesh faces...")
                g_means, g_normals, g_quats, g_scales = self._compute_gaussians_from_mesh(vertices, faces.verts_idx)
                
                ply_path = os.path.join(args.logging_dir, args.expt_name, "gaussians_generated.ply")
                self._save_gaussians_ply(ply_path, g_means, g_normals, g_quats, g_scales)
                self.log.info(f"Saved generated Gaussians to {ply_path}")

            center = vertices.mean(0, keepdim=True)
            vertices-= center
            self.mesh_center = center

            max_coord = vertices.abs().max()
            scale = 1.0 / (max_coord * 1.1)
            vertices*= scale
            self.mesh_scale = scale

            self.template_mesh = Meshes(verts=[vertices], faces=[faces.verts_idx], textures=texture)

            if args.reference_geometry_type == 'gaussians':
                self.gaussian_means_norm = ((g_means - center) * scale).to(device)
                self.gaussian_scales = (g_scales * scale).to(device)
                self.gaussian_normals = g_normals.to(device)
            
            if args.reference_geometry_type == 'udf':
                self.setup_o3d_scene(vertices, faces.verts_idx)

            mesh_name = os.path.basename(args.reference_geometry_source).split(".")[0]
            mesh_outfile = os.path.join(args.logging_dir, args.expt_name, mesh_name + "_t.obj")
            save_obj(mesh_outfile, self.template_mesh.verts_packed(), self.template_mesh.faces_packed())

            if 'mesh_vertices' in args.boundary_condition_name:
                self.boundary_rest_points = (vertices[args.reference_boundary_vertices]) / self.mesh_scale + self.mesh_center
                self.boundary_rest_points_unnorm = vertices[args.reference_boundary_vertices].to(device)
                self.boundary_rest_points = self.boundary_rest_points.to(device)

            if args.reference_geometry_type == 'udf':
                weights_pth = os.path.join(args.logging_dir, args.expt_name, 'weights', 'fitted_mesh.pth')
                self.fit_reference_mlp(args.reference_mlp_lrate, args.reference_mlp_n_iterations, weights_pth=weights_pth, use_pretrained=args.use_pretrained_ref)
                reference_mlp_verts_pred = self(self.template_mesh.verts_padded())
                self.template_mesh = self.template_mesh.update_padded(reference_mlp_verts_pred)
                
                self.setup_o3d_scene(reference_mlp_verts_pred.squeeze(0), self.template_mesh.faces_packed())
            
            self.mesh_scale = self.mesh_scale.to(device)
            self.mesh_center = self.mesh_center.to(device)
            self.template_mesh = self.template_mesh.to(device)

            if tb.writer:
                tb.writer.add_mesh('reference_state', self.template_mesh.verts_padded(), faces=self.template_mesh.faces_padded())
            save_obj(os.path.join(args.logging_dir, args.expt_name, 'reference_state.obj'), self.template_mesh.verts_packed(), self.template_mesh.faces_packed())
        else:
            if domain_bounds is None:
                raise ValueError("domain_bounds are required for analytical reference geometries.")
            args.train_n_spatial_samples, args.test_n_spatial_samples = math.isqrt(args.train_n_spatial_samples) ** 2, math.isqrt(args.test_n_spatial_samples) ** 2
            test_spatial_sidelen = math.isqrt(args.test_n_spatial_samples)
            rest_points = get_mgrid((test_spatial_sidelen, test_spatial_sidelen), stratified=False, dim=2)[None]
            rest_points[...,0] *= domain_bounds.u_max
            rest_points[...,1] *= domain_bounds.v_max
            vertices = self(rest_points)[0]
            faces = torch.tensor(generate_mesh_topology(test_spatial_sidelen), device=device)
            texture = TexturesUV(maps=torch.empty(1, 1, 1, 1, device=device), faces_uvs=[faces], verts_uvs=rest_points)
            self.template_mesh = Meshes(verts=[vertices], faces=[faces], textures=texture).to(device)
            if tb.writer:
                tb.writer.add_mesh('reference_state', self.template_mesh.verts_padded(), faces=self.template_mesh.textures.faces_uvs_padded())
            save_obj(os.path.join(args.logging_dir, args.expt_name, 'reference_state.obj'), self.template_mesh.verts_packed(), self.template_mesh.textures.faces_uvs_padded()[0], verts_uvs=self.template_mesh.textures.verts_uvs_padded()[0], faces_uvs=self.template_mesh.textures.faces_uvs_padded()[0])

    def _compute_gaussians_from_mesh(self, vertices, faces):
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]

        means = (v0 + v1 + v2) / 3.0

        edge1 = v1 - v0
        edge2 = v2 - v0
        cross = torch.cross(edge1, edge2, dim=1)
        
        areas = 0.5 * torch.norm(cross, dim=1, p=2)
        
        normals = torch.nn.functional.normalize(cross, dim=1)

        scale_base = torch.sqrt(areas + 1e-8)
        scales = torch.stack([scale_base, scale_base, scale_base * 0.05], dim=1)

        z_axis = normals
        
        up_vector = torch.tensor([0.0, 1.0, 0.0], device=vertices.device).unsqueeze(0).repeat(len(faces), 1)
        
        dot_product = torch.abs(torch.sum(z_axis * up_vector, dim=1))
        mask = dot_product > 0.99
        up_vector[mask] = torch.tensor([1.0, 0.0, 0.0], device=vertices.device)
        
        x_axis = torch.cross(up_vector, z_axis, dim=1)
        x_axis = torch.nn.functional.normalize(x_axis, dim=1)
        
        y_axis = torch.cross(z_axis, x_axis, dim=1)

        R = torch.stack([x_axis, y_axis, z_axis], dim=2)
        
        quaternions = matrix_to_quaternion(R)

        return means, normals, quaternions, scales

    def _save_gaussians_ply(self, path, positions, normals, quaternions, scales):
        xyz = positions.detach().cpu().numpy()
        nxyz = normals.detach().cpu().numpy()
        rot = quaternions.detach().cpu().numpy()
        
        scale_log = torch.log(scales + 1e-8).detach().cpu().numpy()
        
        opacities = np.ones(len(xyz))
        opacity_logit = np.log(opacities / (1 - opacities + 1e-8))

        dtype = [
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4'),
            ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
            ('opacity', 'f4'),
            ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4') # SH DC (Color)
        ]

        elements = np.zeros(len(xyz), dtype=dtype)
        elements['x'], elements['y'], elements['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        elements['nx'], elements['ny'], elements['nz'] = nxyz[:, 0], nxyz[:, 1], nxyz[:, 2]
        elements['rot_0'] = rot[:, 0]
        elements['rot_1'] = rot[:, 1]
        elements['rot_2'] = rot[:, 2]
        elements['rot_3'] = rot[:, 3]
        elements['scale_0'] = scale_log[:, 0]
        elements['scale_1'] = scale_log[:, 1]
        elements['scale_2'] = scale_log[:, 2]
        elements['opacity'] = opacity_logit
        elements['f_dc_0'] = elements['f_dc_1'] = elements['f_dc_2'] = 0.0

        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def setup_o3d_scene(self, vertices: torch.Tensor, faces: torch.Tensor):
        verts_np = vertices.detach().cpu().numpy().astype(np.float32)
        faces_np = faces.detach().cpu().numpy().astype(np.int32)
        
        mesh_o3d = o3d.geometry.TriangleMesh()
        mesh_o3d.vertices = o3d.utility.Vector3dVector(verts_np)
        mesh_o3d.triangles = o3d.utility.Vector3iVector(faces_np)
        mesh_o3d.compute_vertex_normals()
        
        mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh_o3d)
        
        self.o3d_scene = o3d.t.geometry.RaycastingScene()
        self.o3d_scene.add_triangles(mesh_t)
    
    def compute_distance(self, points: torch.Tensor) -> torch.Tensor:
        if self.o3d_scene is None:
            raise ValueError("Open3D scene not initialised")
        
        points_o3d = torch_to_o3c(points.to(torch.float32))

        signed_dists = self.o3d_scene.compute_signed_distance(points_o3d)

        return o3c_to_torch(signed_dists).abs()

    def sample_training_data(self,
                             samples_on_surface: int,
                             samples_off_surface: int,
                             domain_bounds=([-1, -1, -1], [1, 1, 1])):
        
        surface_points, surface_normals = sample_points_from_meshes(
            self.template_mesh, samples_on_surface, return_normals=True
        )
        surface_points, surface_normals = surface_points[0], surface_normals[0]

        samples_far = samples_off_surface // 2
        samples_near = samples_off_surface - samples_far

        high = torch.tensor(domain_bounds[1])
        low = torch.tensor(domain_bounds[0])
        domain_points = torch.rand((samples_far, 3)) * (high - low) + low

        domain_dists = self.compute_distance(domain_points)

        idx = torch.randint(0, samples_on_surface, (samples_near,))
        pts_subset = surface_points[idx]
        nrm_subset = surface_normals[idx]

        noise = torch.normal(0, 0.01, (samples_near, 1))
        close_points = pts_subset + nrm_subset * noise
        close_dists = self.compute_distance(close_points)

        all_points = torch.cat([surface_points, domain_points, close_points], dim=0)
        all_normals = torch.cat(
            [surface_normals, torch.zeros((samples_far + samples_near, 3))], dim=0
        )
        all_dists = torch.cat(
            [torch.zeros(samples_on_surface), domain_dists, close_dists], dim=0
        ).unsqueeze(1)

        return all_points.float().unsqueeze(0), all_normals.float().unsqueeze(0), all_dists.float().unsqueeze(0)


    def fit_reference_mlp(self, reference_mlp_lrate: float, reference_mlp_n_iterations: int,
                        weights_pth: str, use_pretrained: bool):

        self.reference_mlp = SirenGarment(
            in_features=3,
            out_features=1,
            hidden_layer_config=[256, 256, 256, 256, 256, 256, 256, 256],
            w0=30
        ).to(device)

        if os.path.exists(weights_pth) and use_pretrained:
            self.log.info(f"Loading pretrained weights: {weights_pth}")
            self.reference_mlp.load_state_dict(torch.load(weights_pth))
            return
        
        self.log.info(f"Training Reference MLP in {self.rep_type} mode")

        samples_on_surface = 10000
        samples_off_surface = 20000

        if self.rep_type == "SDF":
            reference_optimizer = torch.optim.Adam(
                lr=reference_mlp_lrate, params=self.reference_mlp.parameters()
            )

            for i in trange(reference_mlp_n_iterations):
                reference_optimizer.zero_grad()

                all_points, surface_normals, all_sdf_gt = self.sample_training_data(
                    samples_on_surface=samples_on_surface,
                    samples_off_surface=samples_off_surface
                )

                all_points = all_points.to(device)
                all_sdf_gt = all_sdf_gt.to(device)
                surface_normals = surface_normals.to(device)

                sdf_pred = self.reference_mlp(all_points)['model_out']
                sdf_loss = torch.mean((sdf_pred - all_sdf_gt) ** 2)

                all_points.requires_grad_(True)
                sdf_pred_grad = self.reference_mlp(all_points)['model_out']
                gradients, _ = jacobian(sdf_pred_grad, all_points)
                gradients = gradients.squeeze(2)
                eikonal_loss = torch.mean((torch.norm(gradients, dim=-1) - 1) ** 2)

                surface_points_detached = all_points.detach().requires_grad_(True)
                surface_sdf = self.reference_mlp(surface_points_detached)['model_out']
                surface_grads, _ = jacobian(surface_sdf, surface_points_detached)
                surface_grads = surface_grads.squeeze(2)
                surface_grads_normalized = surface_grads / (torch.norm(surface_grads, dim=-1, keepdim=True) + 1e-6)
                normal_loss = torch.mean((1 - torch.sum(surface_grads_normalized * surface_normals, dim=-1)) ** 2)

                loss = sdf_loss + 0.1 * eikonal_loss + 0.05 * normal_loss
                loss.backward()
                reference_optimizer.step()

                if tb.writer and i % 100 == 0:
                    tb.writer.add_scalar('loss/sdf', sdf_loss.item(), i)
                    tb.writer.add_scalar('loss/eikonal', eikonal_loss.item(), i)
                    tb.writer.add_scalar('loss/normal', normal_loss.item(), i)
                    tb.writer.add_scalar('loss/total', loss.item(), i)
                    print(f"[SDF] Iter {i}: loss={loss.item():.6f}")

            torch.save(self.reference_mlp.state_dict(), weights_pth)
            self.log.info(f"Reference MLP weights saved: {weights_pth}")

        if self.rep_type == "DUDF":
            use_surrogate = True

            if not use_surrogate:
                reference_optimizer = torch.optim.Adam(lr=reference_mlp_lrate, params=self.reference_mlp.parameters())
            else:
                warmup_epochs = 1000
                s1_epochs = 2000
                warmup_lr = 1e-4
                s1_lr = 1e-5
                s2_lr = 1e-7
                loss_s1_weights = (1e4, 1e4, 1e4, 1e3)
                loss_s2_weights = (1e5, 1e5)
                alpha = 100
                reference_optimizer = torch.optim.Adam(lr=warmup_lr, params=self.reference_mlp.parameters())

                current_loss_weights = loss_s1_weights
                use_loss_s2_current = not use_surrogate

            def loss_s1(model, model_input, gt, loss_weights, alpha):
                model_output = model(model_input)
                udf = gt['sdf']
                gt_normals = gt['normals']
                coords = model_output['model_in']
                pred_sdf = model_output['model_out']
                tan = torch.tanh(alpha * udf)
                tdf = udf * tan
                grad_sdf = gradient(pred_sdf, coords)
                grad_constraint = torch.abs(
                    torch.linalg.norm(grad_sdf.squeeze(0), dim=-1)
                    - torch.abs(tan + udf * alpha * (1 - tan ** 2)).squeeze(-1)
                )
                hessians, _ = hessian(pred_sdf.squeeze(-1), coords)
                _, eigenvectors = torch.linalg.eigh(hessians)
                pred_normals = eigenvectors[..., 2]
                surface_points_mask = torch.flatten(udf == 0)
                principal_direction_constraint = torch.where(
                    surface_points_mask,
                    (1 - torch.abs(F.cosine_similarity(gt_normals, pred_normals, dim=-1))),
                    torch.zeros_like(surface_points_mask)
                )
                on_surf_vals = torch.where(udf == 0, torch.abs(pred_sdf), torch.zeros_like(pred_sdf))
                off_surf_vals = torch.where(udf != 0, torch.abs(tdf - pred_sdf), torch.zeros_like(pred_sdf))
                return {
                    'sdf_on_surf': on_surf_vals.mean() * loss_weights[0],
                    'sdf_off_surf': off_surf_vals.mean() * loss_weights[1],
                    'hessian_constraint': principal_direction_constraint.mean() * loss_weights[2],
                    'grad_constraint': grad_constraint.mean() * loss_weights[3]
                }

            def loss_s2(model, model_input, gt, loss_weights):
                model_output = model(model_input)
                udf = gt['sdf']
                pred_sdf = model_output['model_out']
                std_on_surf = torch.std(pred_sdf[udf == 0])
                mean_on_surf = torch.abs(torch.mean(pred_sdf[udf == 0]))
                return {
                    'sdf_on_surf': mean_on_surf * loss_weights[0],
                    'std_on_surf': std_on_surf * loss_weights[1]
                }

            for i in trange(reference_mlp_n_iterations):
                if use_surrogate:
                    if i == warmup_epochs:
                        for g in reference_optimizer.param_groups:
                            g['lr'] = s1_lr
                        self.log.info(f"Warmup complete. Switching to s1 LR: {s1_lr}")

                    if i == s1_epochs:
                        use_loss_s2_current = True
                        current_loss_weights = loss_s2_weights
                        for g in reference_optimizer.param_groups:
                            g['lr'] = s2_lr
                        self.log.info("Switching to loss_s2 (surface refinement)")

                reference_optimizer.zero_grad()
                all_points, surface_normals, all_sdf_gt = self.sample_training_data(
                    samples_on_surface=samples_on_surface,
                    samples_off_surface=samples_off_surface
                )

                all_points = all_points.to(device)
                all_sdf_gt = all_sdf_gt.to(device)
                surface_normals = surface_normals.to(device)

                gt = {'sdf': all_sdf_gt, 'normals': surface_normals}
                if use_loss_s2_current:
                    losses = loss_s2(self.reference_mlp, all_points, gt, current_loss_weights)
                else:
                    losses = loss_s1(self.reference_mlp, all_points, gt, current_loss_weights, alpha)
                
                loss = sum(losses.values())
                loss.backward()
                reference_optimizer.step()

                if tb.writer and i % 100 == 0:
                    for k, v in losses.items():
                        tb.writer.add_scalar(f'loss/{k}', v.item(), i)
                    tb.writer.add_scalar('loss/total', loss.item(), i)
                    print(f"[DUDF] Iter {i}: loss={loss.item():.6f}")

            torch.save(self.reference_mlp.state_dict(), weights_pth)
            self.log.info(f"Reference MLP weights saved: {weights_pth}")

        if self.rep_type == "UDF":
            reference_optimizer = torch.optim.Adam(
                lr=reference_mlp_lrate, params=self.reference_mlp.parameters()
            )

            for i in trange(reference_mlp_n_iterations):
                reference_optimizer.zero_grad()
                all_points, surface_normals, all_sdf_gt = self.sample_training_data(
                    samples_on_surface=samples_on_surface,
                    samples_off_surface=samples_off_surface
                )

                all_points = all_points.to(device)
                all_sdf_gt = all_sdf_gt.to(device)
                surface_normals = surface_normals.to(device)

                udf_pred = F.softplus(self.reference_mlp(all_points)['model_out'])
                udf_gt = torch.abs(all_sdf_gt)

                udf_loss = torch.mean((udf_pred - udf_gt) ** 2)

                all_points.requires_grad_(True)
                udf_grad = F.softplus(self.reference_mlp(all_points)['model_out'])
                gradients, _ = jacobian(udf_grad, all_points)
                gradients = gradients.squeeze(2)
                eikonal_loss = torch.mean((torch.clamp_min(torch.norm(gradients, dim=-1) - 1, 0)) ** 2)

                loss = udf_loss + 0.01 * eikonal_loss
                loss.backward()
                reference_optimizer.step()

                if tb.writer and i % 100 == 0:
                    tb.writer.add_scalar('loss/udf', udf_loss.item(), i)
                    tb.writer.add_scalar('loss/eikonal', eikonal_loss.item(), i)
                    tb.writer.add_scalar('loss/total', loss.item(), i)
                    print(f"[UDF] Iter {i}: loss={loss.item():.6f}")

            torch.save(self.reference_mlp.state_dict(), weights_pth)
            self.log.info(f"Reference MLP weights saved: {weights_pth}")

    def level_set_function(self, x):
        original_shape = x.shape
        is_batched = len(x.shape) == 3
        
        if is_batched:
            x = x.view(-1, 3)
        
        x0, x1, x2 = x[:, 0], x[:, 1], x[:, 2]

        match self.reference_geometry_name:
            case 'rectangle_xy':
                phi = x2 
            case 'rectangle_xz':
                phi = x1 
            case 'cylinder':
                R = 0.25
                phi = x0**2 + x2**2 - R**2
            case 'cone':
                R_top, R_bottom, L = 0.2, 1.5, 1
                radius_at_y = R_bottom + (R_top - R_bottom) * x[:,1] / L
                phi = x0**2 + x2**2 - radius_at_y**2
            case 'mesh':
                if self.reference_mlp is None:
                    raise ValueError("reference_mlp must be provided for mesh geometry")
                phi = self.reference_mlp(x)['model_out']
            case _:
                raise ValueError(f"Unknown reference_geometry_name {self.reference_geometry_name}")
        
        if is_batched:
            B, N, _ = original_shape
            phi = phi.view(B, N, 1)
        else:
            phi = phi.unsqueeze(-1)
        
        return phi

    def __call__(self, coords: torch.Tensor) -> torch.Tensor:

        if self.reference_geometry_name == 'mesh': return coords

        u = coords[...,0] 
        v = coords[...,1]
        match self.reference_geometry_name:
            case 'rectangle_xy': #vertical
                midsurface_positions = torch.stack([u, v, 0.* (u**2 - v**2)], dim=2)
            case 'rectangle_xz': #horizontal
                midsurface_positions = torch.stack([u, 0.* (u**2 - v**2), v], dim=2)
            case 'cylinder':
                R = 0.25
                midsurface_positions = torch.stack([R * torch.cos(u), v, R * torch.sin(u)], dim=2)
            case 'cone':
                R_top, R_bottom, L = 0.2, 1.5, 1
                R = v * (R_top - R_bottom) / L + R_bottom
                midsurface_positions = torch.stack([R * torch.cos(u), v, R * torch.sin(u)], dim=2)
            case _: 
                raise ValueError(f'Unknown reference_geometry_name {self.reference_geometry_name}')
        return midsurface_positions
