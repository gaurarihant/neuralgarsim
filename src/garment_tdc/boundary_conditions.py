import math
import torch

class BoundaryCondition:
    def __init__(
        self,
        reference_geometry_name: str,
        boundary_condition_name: str,
        boundary_points: torch.Tensor = None,
        mesh_verts = None,
        moved_boundary_mode: str = "away",
        moved_boundary_displacement: float = 0.05,
    ):
        self.boundary_condition_name = boundary_condition_name
        self.reference_geometry_name = reference_geometry_name
        self.boundary_points = boundary_points
        self.boundary_support = 0.01
        self.moved_boundary_mode = moved_boundary_mode
        self.moved_boundary_displacement = moved_boundary_displacement

        if reference_geometry_name == "mesh" and mesh_verts is not None:
            self.min_vals = mesh_verts.amin(dim=1, keepdim=True).detach()
            self.max_vals = mesh_verts.amax(dim=1, keepdim=True).detach()
        else:
            self.min_vals, self.max_vals = None, None
    
    def normalization(self, points: torch.Tensor) -> torch.Tensor:
        match self.reference_geometry_name:
            case 'rectangle_xy':
                normalized_coords = torch.cat([
                    (points[..., 0:1]), 
                    (points[..., 1:2]),
                    points[..., 2:3]
                ], dim=2)
            case 'rectangle_xz':
                normalized_coords = torch.cat([
                    (points[..., 0:1]), 
                    points[..., 1:2],
                    (points[..., 2:3])
                ], dim=2)
            case 'cylinder':
                xi1 = torch.atan2(points[..., 2:3], points[..., 0:1])
                xi2 = points[..., 1:2]
                
                normalized_coords = torch.cat([
                    (torch.cos(xi1) + 1)/2,
                    (torch.sin(xi1) + 1)/2,
                    xi2
                ], dim=2)
            case 'cone':
                R_top, R_bottom, L = 0.2, 1.5, 1
                y_norm = points[..., 1:2] / L
                max_R = points[..., 1:2] * (R_top - R_bottom) / L + R_bottom
                x_norm = (points[..., 0:1] / max_R + 1) / 2
                z_norm = (points[..., 2:3] / max_R + 1) / 2
                normalized_coords = torch.cat([x_norm, y_norm, z_norm], dim=2)
            case 'mesh':
                normalized_coords = (points - self.min_vals) / (self.max_vals - self.min_vals + 1e-12)
            case _:
                min_vals = points.amin(dim=1, keepdim=True) 
                max_vals = points.amax(dim=1, keepdim=True)
                normalized_coords = (points - min_vals) / (max_vals - min_vals + 1e-12)
        
        return normalized_coords

    def dirichlet_condition(self, deformations: torch.Tensor, points: torch.Tensor) -> torch.Tensor:   

        min_coords, _ = torch.min(points, dim=1, keepdim=True)
        max_coords, _ = torch.max(points, dim=1, keepdim=True)

        x_min, y_min, z_min = min_coords.split(1, dim=-1)
        x_max, y_max, z_max = max_coords.split(1, dim=-1)

        x, y, z = points.split(1, dim=-1)

        match self.boundary_condition_name:
            case 'top_left_fixed':
                top_left_corner = torch.cat([x_min, y_max, z_max], dim=-1)                
                top_left_corner = torch.exp(-torch.sum((points - top_left_corner)**2, dim=-1, keepdim=True) / self.boundary_support)
                deformations = deformations * (1 - top_left_corner)
            case 'top_left_top_right_moved':
                top_left_corner = torch.cat([x_min, y_max, z_min], dim=-1)
                top_left_corner = torch.exp(-torch.sum((points - top_left_corner)**2, dim=-1, keepdim=True) / self.boundary_support)

                top_right_corner = torch.cat([x_max, y_max, z_min], dim=-1)
                top_right_corner = torch.exp(-torch.sum((points - top_right_corner)**2, dim=-1, keepdim=True) / self.boundary_support)

                temporal_motion = 0.2 * torch.ones_like(x)               
                corner_displacement = torch.cat([temporal_motion, torch.zeros_like(temporal_motion), torch.zeros_like(temporal_motion)], dim=-1)
                
                deformations = deformations * (1 - top_left_corner) * (1 - top_right_corner) + corner_displacement * top_left_corner - corner_displacement * top_right_corner                                
            case 'adjacent_edges_fixed':
                left_edge = torch.exp(-(x ** 2)/self.boundary_support)
                right_edge = torch.exp(-(z ** 2)/self.boundary_support)
                deformations = deformations * (1 - left_edge) * (1 - right_edge)
            case 'nonboundary_handle_fixed':
                center_point = torch.exp(-((x - 0.5)**2 + (y)**2 + (z - 0.7)**2)/self.boundary_support)
                deformations = deformations * (1 - center_point)
            case 'nonboundary_edge_fixed':
                center_edge = torch.exp(-((x - 0.5) ** 2)/self.boundary_support)
                deformations = deformations * (1 - center_edge)
            case 'top_bottom_rims_compression':  
                self.boundary_support = 0.01              
                bottom_rim = torch.exp(-(y ** 2)/self.boundary_support)
                top_rim = torch.exp(-((y - 1.0) ** 2)/self.boundary_support)
                temporal_motion = 0.075 * torch.ones_like(y)
                rim_displacement = torch.cat([torch.zeros_like(temporal_motion), temporal_motion, torch.zeros_like(temporal_motion)], dim=-1)
                deformations = deformations * (1 - bottom_rim) * (1 - top_rim) - rim_displacement * top_rim + rim_displacement * bottom_rim
            case 'bottom_rims_compression':  
                self.boundary_support = 0.01              
                bottom_rim = torch.exp(-(y ** 2)/self.boundary_support)
                temporal_motion = 0.075 * torch.ones_like(y)
                rim_displacement = torch.cat([torch.zeros_like(temporal_motion), temporal_motion, torch.zeros_like(temporal_motion)], dim=-1) #2
                deformations = deformations * (1 - bottom_rim) + rim_displacement * bottom_rim
            case 'top_bottom_rims_torsion':
                self.boundary_support = 0.01
                bottom_rim = torch.exp(-(y ** 2)/self.boundary_support)
                top_rim = torch.exp(-((y - 1.0) ** 2)/self.boundary_support)
                R = 0.25
                rotation = math.pi / 4 #3 * math.pi / 4

                temporal_motion = torch.ones_like(x) * rotation

                center_x = 0.5 * (x_min + x_max)
                center_z = 0.5 * (z_min + z_max)

                rel_x, rel_z = x - center_x, z - center_z
                theta = torch.atan2(rel_z, rel_x)
                radius = torch.sqrt(rel_x**2 + rel_z**2)

                top_theta_new = theta + temporal_motion
                top_x_new = center_x + radius * torch.cos(top_theta_new)
                top_z_new = center_z + radius * torch.sin(top_theta_new)

                bottom_theta_new = theta - temporal_motion
                bottom_x_new = center_x + radius * torch.cos(bottom_theta_new)
                bottom_z_new = center_z + radius * torch.sin(bottom_theta_new)

                top_rim_displacement = torch.cat([top_x_new - x, torch.zeros_like(y), top_z_new - z], dim=-1)
                bottom_rim_displacement = torch.cat([bottom_x_new - x, torch.zeros_like(y), bottom_z_new - z], dim=-1)
                
                deformations = deformations * (1 - bottom_rim) * (1 - top_rim) + top_rim_displacement * top_rim + bottom_rim_displacement * bottom_rim              
            case 'top_rim_fixed':
                self.boundary_support = 0.01
                top_rim = torch.exp(-((y - 1.0) ** 2)/self.boundary_support)
                deformations = deformations * (1 - top_rim)
            case 'top_rim_torsion':
                self.boundary_support = 0.001
                top_rim = torch.exp(-((y - 1.0) ** 2)/self.boundary_support)
                
                R_top_val = 0.2
                
                theta = torch.atan2(z, x)
                rotation = math.pi / 2
                temporal_motion = torch.ones_like(x) * rotation
                
                top_rim_displacement = torch.cat([
                    R_top_val * (torch.cos(theta + temporal_motion) - torch.cos(theta)),
                    torch.zeros_like(temporal_motion),
                    R_top_val * (torch.sin(theta + temporal_motion) - torch.sin(theta))
                ], dim=2)
                
                deformations = deformations * (1 - top_rim) + top_rim_displacement * top_rim
            case 'mesh_vertices':
                if self.boundary_points is not None:
                    for i in range(self.boundary_points.shape[0]):
                        boundary_point = torch.exp(-torch.sum((points - self.boundary_points[i])**2, dim=-1, keepdim=True) / self.boundary_support)
                        deformations = deformations * (1 - boundary_point)
            case 'mesh_vertices_moved':
                if self.boundary_points is not None:
                    if self.boundary_points.shape[0] != 2:
                        raise ValueError("mesh_vertices_moved expects exactly two reference_boundary_vertices.")

                    handle_axis = self.boundary_points[1] - self.boundary_points[0]
                    handle_axis = handle_axis / handle_axis.norm().clamp_min(1e-12)
                    first_sign, second_sign = (-1.0, 1.0)
                    if self.moved_boundary_mode == "towards":
                        first_sign, second_sign = (1.0, -1.0)
                    elif self.moved_boundary_mode != "away":
                        raise ValueError("moved_boundary_mode must be either 'away' or 'towards'.")

                    point_displacement_0 = first_sign * self.moved_boundary_displacement * handle_axis.view(1, 1, 3)
                    point_displacement_1 = second_sign * self.moved_boundary_displacement * handle_axis.view(1, 1, 3)
                    boundary_point_mask_0 = torch.exp(-torch.sum((points - self.boundary_points[0])**2, dim=-1, keepdim=True) / self.boundary_support)
                    boundary_point_mask_1 = torch.exp(-torch.sum((points - self.boundary_points[1])**2, dim=-1, keepdim=True) / self.boundary_support)
                    deformations = (
                        deformations * (1 - boundary_point_mask_0) * (1 - boundary_point_mask_1)
                        + point_displacement_0 * boundary_point_mask_0
                        + point_displacement_1 * boundary_point_mask_1
                    )
            case 'free_fall':
                pass
            case _:
                raise ValueError(f'Unknown boundary condition: {self.boundary_condition_name}')
        return deformations
