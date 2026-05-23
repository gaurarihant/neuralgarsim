import torch
from torch.nn.functional import normalize

from .differential_operators import jacobian, hessian
from .rest_surface import RestSurface

class SurfaceGeometry():
    def __init__(self, reference_midsurface: RestSurface, n_spatial_samples: int):
        self.n_spatial_samples = n_spatial_samples 
        self.reference_midsurface = reference_midsurface
    
    def __call__(self, rest_points: torch.Tensor, debug: bool):
        raise NotImplementedError

class SurfaceGeometryGaussians(SurfaceGeometry):
    def __init__(self, reference_midsurface: RestSurface, n_spatial_samples: int):
        super().__init__(reference_midsurface, n_spatial_samples)

        self.means = self.reference_midsurface.gaussian_means_norm
        self.cloud_normals = self.reference_midsurface.gaussian_normals

        scales = self.reference_midsurface.gaussian_scales
        self.cloud_areas = scales[:, 0] * scales[:, 1]

    def nearest_gaussian(self, points: torch.Tensor):
        pts = points.detach()
        B, N, _ = pts.shape
        means = self.means.unsqueeze(0).expand(B, -1, -1)  # [B, M, 3]
        chunk, out = 256, []
        for b in range(B):
            row = []
            for s in range(0, N, chunk):
                d = torch.cdist(pts[b, s:s + chunk], means[b])
                row.append(d.argmin(dim=1))
            out.append(torch.cat(row))
        return torch.stack(out)  # [B, N]

    def gather_gaussian_attr(self, attr: torch.Tensor, idx: torch.Tensor):
        B, N = idx.shape
        return torch.gather(
            attr.unsqueeze(0).expand(B, -1, -1), 1,
            idx.unsqueeze(-1).expand(-1, -1, attr.shape[-1])
        )

    def get_tangents(self):
        batch_size, n_points, _ = self.points.shape
        reference = torch.tensor([1.0, 0.0, 0.0], device=self.points.device).view(1, 1, 3).expand(batch_size, n_points, 3)
        mask = (torch.abs(self.normals[..., 0]) > 0.9).unsqueeze(-1)
        alternative = torch.tensor([0.0, 1.0, 0.0], device=self.points.device).view(1, 1, 3).expand(batch_size, n_points, 3)

        reference = torch.where(mask, alternative, reference)

        self.a_1 = normalize(torch.cross(self.normals, reference, dim=2), dim=2)
        self.a_2 = normalize(torch.cross(self.normals, self.a_1, dim=2), dim=2)
    
    def get_weingarten(self):
        batch_size, n_points, _ = self.points.shape
        self.weingarten = torch.zeros((batch_size, n_points, 3, 3), device=self.points.device)
    
    def __call__(self, rest_points: torch.Tensor, debug: bool):
        self.rest_points = rest_points
        self.rest_points.requires_grad_(True)

        points_norm = self.reference_midsurface(self.rest_points)

        gaussian_idx = self.nearest_gaussian(points_norm)
        self.normals = normalize(self.gather_gaussian_attr(self.cloud_normals, gaussian_idx), dim=-1)

        scale  = self.reference_midsurface.mesh_scale
        center = self.reference_midsurface.mesh_center
        self.points = (points_norm / scale) + center  # world space

        identity = torch.eye(3, device=self.points.device).view(1, 1, 3, 3)
        n  = self.normals.unsqueeze(-1)
        nT = self.normals.unsqueeze(-2)
        self.projector = identity - n @ nT

        self.get_weingarten()
        self.get_tangents()

        self.area_element = self.gather_gaussian_attr(
            self.cloud_areas.unsqueeze(-1), gaussian_idx
        ).squeeze(-1)  # [B, N]

class SurfaceGeometryUDF(SurfaceGeometry): 
    def __init__(self, reference_midsurface: RestSurface, n_spatial_samples: int):
        super().__init__(reference_midsurface, n_spatial_samples)

    def get_normal(self):
        phi = self.reference_midsurface.level_set_function(self.points).unsqueeze(-1)
        gradients, status = jacobian(phi, self.points)
        gradients = gradients.squeeze(2)

        self.grad_phi_norm = gradients.norm(dim=-1, keepdim=True)
        normals = gradients / self.grad_phi_norm

        self.normals = normals

    def get_tangents(self):
        batch_size, n_points, _ = self.points.shape
        reference = torch.tensor([1.0, 0.0, 0.0], device=self.points.device).view(1, 1, 3).expand(batch_size, n_points, 3)
        mask = (torch.abs(self.normals[..., 0]) > 0.9).unsqueeze(-1)
        alternative = torch.tensor([0.0, 1.0, 0.0], device=self.points.device).view(1, 1, 3).expand(batch_size, n_points, 3)

        reference = torch.where(mask, alternative, reference)

        self.a_1 = normalize(torch.cross(self.normals, reference, dim=2), dim=2)
        self.a_2 = normalize(torch.cross(self.normals, self.a_1, dim=2), dim=2)

    def get_weingarten(self):
        batch_size, n_points, _ = self.points.shape
        
        if self.reference_midsurface.reference_geometry_name in ['rectangle_xy', 'rectangle_xz']:
            self.weingarten = torch.zeros((batch_size, n_points, 3, 3), device=self.points.device)
            H_phi = None
        else:        
            phi = self.reference_midsurface.level_set_function(self.points)

            H_phi, status = hessian(phi, self.points)
            if status != 0:
                raise ValueError("NaN in Hessian of level set function.")

            if self.reference_midsurface.reference_geometry_name == 'mesh':
                _, eigenvectors = torch.linalg.eigh(H_phi)
                gradients = eigenvectors[..., 2]
                self.grad_phi_norm = gradients.norm(dim=-1, keepdim=True)
                self.normals = gradients / self.grad_phi_norm
        
        identity = torch.eye(3, device=self.points.device).view(1, 1, 3, 3)
        n = self.normals.unsqueeze(-1)
        nT = self.normals.unsqueeze(-2)
        self.projector = identity - n @ nT

        if H_phi is not None:   
            H_phi = H_phi.squeeze(2)
            self.weingarten = -(self.projector @ H_phi @ self.projector) / self.grad_phi_norm.unsqueeze(-1)

    def __call__(self, rest_points: torch.Tensor, debug: bool):
        self.rest_points = rest_points
        self.rest_points.requires_grad_(True)

        self.points = self.reference_midsurface(self.rest_points)
        if self.reference_midsurface.reference_geometry_name != 'mesh':
            self.get_normal()
        
        self.get_weingarten()

        if self.reference_midsurface.reference_geometry_name == 'mesh':
            scale = self.reference_midsurface.mesh_scale
            center = self.reference_midsurface.mesh_center

            self.points = (self.points / scale) + center
            self.weingarten /= scale

        self.get_tangents()
        a_1_1 = torch.einsum('ijk,ijk->ij', self.a_1, self.a_1)
        a_1_2 = torch.einsum('ijk,ijk->ij', self.a_1, self.a_2)
        a_2_2 = torch.einsum('ijk,ijk->ij', self.a_2, self.a_2)
        det_g = a_1_1 * a_2_2 - a_1_2 * a_1_2

        self.area_element = torch.sqrt(det_g)
        


class SurfaceGeometryMesh(SurfaceGeometry):
    def __init__(self, reference_midsurface: RestSurface, n_spatial_samples: int):
        super().__init__(reference_midsurface, n_spatial_samples)

        self.mesh = reference_midsurface.template_mesh

        self._oriented_normals = self._orient_normals(
            self.mesh.verts_packed(),
            self.mesh.verts_normals_packed()
        ).unsqueeze(0)

        self._vertex_areas = self._compute_vertex_areas().unsqueeze(0)

    @staticmethod
    def _orient_normals(verts: torch.Tensor, normals: torch.Tensor):
        centroid = verts.mean(dim=0, keepdim=True)
        to_out   = verts - centroid
        dot      = (normals * to_out).sum(dim=-1)
        flip     = torch.where(dot < 0,
                               torch.full_like(dot, -1.0),
                               torch.ones_like(dot))
        return normalize(normals * flip.unsqueeze(-1), dim=-1)

    def _compute_vertex_areas(self):
        scale = self.reference_midsurface.mesh_scale   # scalar: verts_world = verts_norm / scale + center
        verts = self.mesh.verts_packed()               # [V, 3]  normalised
        faces = self.mesh.faces_packed()               # [F, 3]
        V = verts.shape[0]

        v0 = verts[faces[:, 0]]
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]

        face_areas_norm = 0.5 * torch.cross(v1 - v0, v2 - v0, dim=-1).norm(dim=-1)  # [F]

        face_areas_world = face_areas_norm / (scale ** 2)

        vertex_areas = torch.zeros(V, dtype=verts.dtype, device=verts.device)
        contrib = face_areas_world / 3.0
        vertex_areas.scatter_add_(0, faces[:, 0], contrib)
        vertex_areas.scatter_add_(0, faces[:, 1], contrib)
        vertex_areas.scatter_add_(0, faces[:, 2], contrib)

        return vertex_areas.unsqueeze(-1)  # [V, 1]

    def nearest_vertex(self, points: torch.Tensor):
        pts   = points.detach()
        B, N, _ = pts.shape
        verts = self.mesh.verts_padded().expand(B, -1, -1)
        chunk, out = 512, []
        for s in range(0, N, chunk):
            out.append(torch.cdist(pts[:, s:s + chunk], verts).argmin(dim=2))
        return torch.cat(out, dim=1)  # [B, N]

    def gather_vattr(self, attr: torch.Tensor, idx: torch.Tensor):
        B, N = idx.shape
        return torch.gather(
            attr.expand(B, -1, -1), 1,
            idx.unsqueeze(-1).expand(-1, -1, attr.shape[-1])
        )

    def get_tangents(self):
        batch_size, n_points, _ = self.points.shape
        reference = torch.tensor([1.0, 0.0, 0.0], device=self.points.device).view(1, 1, 3).expand(batch_size, n_points, 3)
        mask = (torch.abs(self.normals[..., 0]) > 0.9).unsqueeze(-1)
        alternative = torch.tensor([0.0, 1.0, 0.0], device=self.points.device).view(1, 1, 3).expand(batch_size, n_points, 3)

        reference = torch.where(mask, alternative, reference)

        self.a_1 = normalize(torch.cross(self.normals, reference, dim=2), dim=2)
        self.a_2 = normalize(torch.cross(self.normals, self.a_1, dim=2), dim=2)

    def get_weingarten(self):
        batch_size, n_points, _ = self.points.shape
        self.weingarten = torch.zeros((batch_size, n_points, 3, 3), device=self.points.device)

    def __call__(self, rest_points: torch.Tensor, debug: bool):
        self.rest_points = rest_points
        self.rest_points.requires_grad_(True)

        points_norm = self.reference_midsurface(self.rest_points)

        vert_idx = self.nearest_vertex(points_norm)
        self.normals = normalize(self.gather_vattr(self._oriented_normals, vert_idx), dim=-1)

        scale  = self.reference_midsurface.mesh_scale
        center = self.reference_midsurface.mesh_center
        self.points = (points_norm / scale) + center  # [B, N, 3] world space

        identity = torch.eye(3, device=self.points.device).view(1, 1, 3, 3)
        n  = self.normals.unsqueeze(-1)
        nT = self.normals.unsqueeze(-2)
        self.projector = identity - n @ nT

        self.get_weingarten()
        self.get_tangents()

        self.area_element = self.gather_vattr(self._vertex_areas, vert_idx).squeeze(-1)  # [B, N]


class SurfaceGeometryPCD(SurfaceGeometry):
    def __init__(self, reference_midsurface: RestSurface, n_spatial_samples: int, k_normals: int = 20):
        super().__init__(reference_midsurface, n_spatial_samples)

        self.k_normals = k_normals
        self.cloud = reference_midsurface.template_mesh.verts_packed()  # [M, 3]

        scale = reference_midsurface.mesh_scale
        mesh  = reference_midsurface.template_mesh
        verts = mesh.verts_packed()
        faces = mesh.faces_packed()
        V = verts.shape[0]

        v0 = verts[faces[:, 0]]
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]
        face_areas_norm  = 0.5 * torch.cross(v1 - v0, v2 - v0, dim=-1).norm(dim=-1)
        face_areas_world = face_areas_norm / (scale ** 2)

        vertex_areas = torch.zeros(V, dtype=verts.dtype, device=verts.device)
        contrib = face_areas_world / 3.0
        vertex_areas.scatter_add_(0, faces[:, 0], contrib)
        vertex_areas.scatter_add_(0, faces[:, 1], contrib)
        vertex_areas.scatter_add_(0, faces[:, 2], contrib)

        self.cloud_areas = vertex_areas  # [M]

        self.cloud_normals = self.estimate_pca_normals(
            self.cloud,
            k=self.k_normals,
            orient=True
        )

    def estimate_pca_normals(self, points: torch.Tensor, k: int = 20, orient: bool = True):
        """
        points: [M, 3]
        returns normals: [M, 3]
        PCA normal = eigenvector of smallest covariance eigenvalue.
        """
        M = points.shape[0]
        k = min(k, M)

        with torch.no_grad():
            dists = torch.cdist(points.unsqueeze(0), points.unsqueeze(0))[0]  # [M, M]
            knn_idx = dists.topk(k=k, largest=False).indices                 # [M, k]

            neigh = points[knn_idx]                                          # [M, k, 3]
            centered = neigh - neigh.mean(dim=1, keepdim=True)

            cov = centered.transpose(1, 2) @ centered / max(k - 1, 1)         # [M, 3, 3]
            eigvals, eigvecs = torch.linalg.eigh(cov)

            normals = eigvecs[..., 0]                                        # smallest eigenvalue
            normals = normalize(normals, dim=-1)

            if orient:
                center = points.mean(dim=0, keepdim=True)
                to_point = points - center
                flip = (normals * to_point).sum(dim=-1, keepdim=True) < 0
                normals = torch.where(flip, -normals, normals)

        return normals

    def nearest_cloud_point(self, points: torch.Tensor):
        pts = points.detach()
        B, N, _ = pts.shape
        chunk, out = 256, []
        for b in range(B):
            row = []
            for s in range(0, N, chunk):
                d = torch.cdist(pts[b, s:s + chunk], self.cloud)
                row.append(d.argmin(dim=1))
            out.append(torch.cat(row))
        return torch.stack(out)  # [B, N]

    def gather_cloud_attr(self, attr: torch.Tensor, idx: torch.Tensor):
        B, N = idx.shape
        return torch.gather(
            attr.unsqueeze(0).expand(B, -1, -1), 1,
            idx.unsqueeze(-1).expand(-1, -1, attr.shape[-1])
        )

    def get_tangents(self):
        batch_size, n_points, _ = self.points.shape
        reference = torch.tensor([1.0, 0.0, 0.0], device=self.points.device).view(1, 1, 3).expand(batch_size, n_points, 3)
        mask = (torch.abs(self.normals[..., 0]) > 0.9).unsqueeze(-1)
        alternative = torch.tensor([0.0, 1.0, 0.0], device=self.points.device).view(1, 1, 3).expand(batch_size, n_points, 3)

        reference = torch.where(mask, alternative, reference)

        self.a_1 = normalize(torch.cross(self.normals, reference, dim=2), dim=2)
        self.a_2 = normalize(torch.cross(self.normals, self.a_1, dim=2), dim=2)

    def get_weingarten(self):
        batch_size, n_points, _ = self.points.shape
        self.weingarten = torch.zeros((batch_size, n_points, 3, 3), device=self.points.device)

    def __call__(self, rest_points: torch.Tensor, debug: bool):
        self.rest_points = rest_points
        self.rest_points.requires_grad_(True)

        points_norm = self.reference_midsurface(self.rest_points)

        cloud_idx = self.nearest_cloud_point(points_norm)
        self.normals = normalize(self.gather_cloud_attr(self.cloud_normals, cloud_idx), dim=-1)

        scale  = self.reference_midsurface.mesh_scale
        center = self.reference_midsurface.mesh_center
        self.points = (points_norm / scale) + center  # world space

        identity = torch.eye(3, device=self.points.device).view(1, 1, 3, 3)
        n  = self.normals.unsqueeze(-1)
        nT = self.normals.unsqueeze(-2)
        self.projector = identity - n @ nT

        self.get_weingarten()
        self.get_tangents()

        self.area_element = self.gather_cloud_attr(
            self.cloud_areas.unsqueeze(-1), cloud_idx
        ).squeeze(-1)  # [B, N]
