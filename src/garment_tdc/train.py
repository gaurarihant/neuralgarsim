import os
from shutil import copyfile

import torch
from torch.utils.data import DataLoader
from tqdm import trange

from . import tb
from .boundary_conditions import BoundaryCondition
from .config import device, get_config_parser
from .energy import LinearMaterial, NonLinearMaterial, ShellEnergy
from .logging_utils import get_logger
from .mesh_io import save_meshes
from .models import SirenDeformationField
from .pipeline import run_mesh_preprocessing, run_postprocessing
from .rest_surface import RestSurface
from .sampling import DomainBounds, GridSampler, MeshSampler, UDFSampler
from .surface_geometry import (
    SurfaceGeometryGaussians,
    SurfaceGeometryMesh,
    SurfaceGeometryPCD,
    SurfaceGeometryUDF,
)

torch.manual_seed(42)


def evaluate(args, deformation_field: SirenDeformationField, rest_surface: RestSurface, meshes_dir: str, step: int):
    """Evaluate the neural deformation field on the template surface and save a mesh."""
    if rest_surface.reference_geometry_name == "mesh":
        mesh_verts = (rest_surface.template_mesh.verts_padded() / rest_surface.mesh_scale) + rest_surface.mesh_center
        test_deformations = deformation_field(mesh_verts)
        test_deformed_positions = mesh_verts + test_deformations
        faces = rest_surface.template_mesh.faces_packed()
        verts_uvs = None
    else:
        test_deformations = deformation_field(rest_surface.template_mesh.verts_padded())
        test_deformed_positions = rest_surface.template_mesh.verts_padded() + test_deformations
        faces = rest_surface.template_mesh.textures.faces_uvs_padded()[0]
        verts_uvs = rest_surface.template_mesh.textures.verts_uvs_padded()[0]

    if tb.writer:
        tb.writer.add_mesh(
            "simulated_states",
            test_deformed_positions,
            faces=rest_surface.template_mesh.faces_padded(),
            global_step=step,
        )

    save_meshes(test_deformed_positions, faces, meshes_dir, step, verts_uvs)


def build_surface_geometry(args, rest_surface: RestSurface):
    if args.reference_geometry_type == "mesh":
        return SurfaceGeometryMesh(rest_surface, args.train_n_spatial_samples)
    if args.reference_geometry_type == "pcd":
        return SurfaceGeometryPCD(rest_surface, args.train_n_spatial_samples)
    if args.reference_geometry_type == "gaussians":
        return SurfaceGeometryGaussians(rest_surface, args.train_n_spatial_samples)
    return SurfaceGeometryUDF(rest_surface, args.train_n_spatial_samples)


def train():
    args = get_config_parser().parse_args()

    log_dir = os.path.join(args.logging_dir, args.expt_name)
    meshes_dir = os.path.join(log_dir, "meshes")
    weights_dir = os.path.join(log_dir, "weights")

    for directory in [log_dir, weights_dir]:
        os.makedirs(directory, exist_ok=True)

    logger = get_logger(log_dir, args.expt_name)
    logger.info(args)
    tb.set_tensorboard_writer(log_dir, args.debug)

    if args.preprocess_mesh:
        run_mesh_preprocessing(args, logger)

    copyfile(args.config_filepath, os.path.join(log_dir, "args.ini"))

    domain_bounds = None if args.reference_geometry_name == "mesh" else DomainBounds(args.domain_u_max, args.domain_v_max)
    rest_surface = RestSurface(args, domain_bounds, logger)
    surface_geometry = build_surface_geometry(args, rest_surface)

    if args.reference_geometry_name == "mesh":
        mesh_verts = (rest_surface.template_mesh.verts_padded() / rest_surface.mesh_scale) + rest_surface.mesh_center
        boundary = BoundaryCondition(
            args.reference_geometry_name,
            args.boundary_condition_name,
            rest_surface.boundary_rest_points,
            mesh_verts,
            args.moved_boundary_mode,
            args.moved_boundary_displacement,
        )
    else:
        boundary = BoundaryCondition(
            args.reference_geometry_name,
            args.boundary_condition_name,
            rest_surface.boundary_rest_points,
            moved_boundary_mode=args.moved_boundary_mode,
            moved_boundary_displacement=args.moved_boundary_displacement,
        )

    material = LinearMaterial(args, surface_geometry) if args.material_type == "linear" else NonLinearMaterial(args, surface_geometry)
    shell_energy = ShellEnergy(
        surface_geometry,
        material,
        args.gravity_acceleration,
        args.i_debug,
    )

    deformation_field = SirenDeformationField(
        boundary,
        in_features=3,
        out_features=3,
        hidden_features=args.hidden_features,
        hidden_layers=args.hidden_layers,
        first_omega_0=args.siren_omega,
        hidden_omega_0=args.siren_omega,
    ).to(device)

    optimizer = torch.optim.Adam(lr=args.lrate, params=deformation_field.parameters())

    if args.i_ckpt is not None:
        ckpts = [os.path.join(weights_dir, f"{args.i_ckpt:06d}.tar")]
    else:
        ckpts = [os.path.join(weights_dir, f) for f in sorted(os.listdir(weights_dir)) if f.endswith(".tar")]

    logger.info(f"Found ckpts: {ckpts}")
    if args.init_deformation_ckpt is not None:
        logger.info(f"Warm-starting deformation field from checkpoint {args.init_deformation_ckpt}")
        ckpt = torch.load(args.init_deformation_ckpt)
        deformation_field.load_state_dict(ckpt["siren_state_dict"], strict=True)
        optimizer = torch.optim.Adam(lr=args.edit_lrate, params=deformation_field.parameters())
        global_step = 0
    elif len(ckpts) > 0 and not args.no_reload:
        logger.info(f"Resuming experiment {args.expt_name} from checkpoint {ckpts[-1]}")
        ckpt = torch.load(ckpts[-1])
        global_step = ckpt["global_step"]
        deformation_field.load_state_dict(ckpt["siren_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    else:
        logger.info(f"Starting experiment {args.expt_name}")
        global_step = 0

    if args.test_only:
        if len(ckpts) == 0:
            raise RuntimeError("test_only=True requires an existing checkpoint.")
        logger.info(f"Evaluating deformation field from checkpoint {ckpts[-1]}")
        evaluate(args, deformation_field, rest_surface, meshes_dir, global_step)
        run_postprocessing(args, log_dir, logger)
        return

    if args.reference_geometry_name == "mesh":
        use_udf_sampler = args.reference_geometry_type == "udf" and args.use_udf
        sampler = UDFSampler(args.train_n_spatial_samples, rest_surface.reference_mlp) if use_udf_sampler else MeshSampler(args.train_n_spatial_samples, rest_surface.template_mesh)
    else:
        if domain_bounds is None:
            raise RuntimeError("Analytical surface sampling requires domain bounds.")
        sampler = GridSampler(args.train_n_spatial_samples, domain_bounds)

    dataloader = DataLoader(sampler, batch_size=1, num_workers=0)

    if tb.writer:
        tb.writer.add_text("args", str(args))

    final_step = global_step
    for step in trange(global_step, args.n_iterations):
        final_step = step
        data = next(iter(dataloader))
        if isinstance(data, (list, tuple)):
            rest_points, sample_weights = data
            sample_weights = sample_weights.to(device)
        else:
            rest_points = data
            sample_weights = 1.0

        surface_geometry(rest_points, step == 0)
        deformations = deformation_field(surface_geometry.points)

        mechanical_energy = shell_energy(deformations, step)
        loss = (mechanical_energy * sample_weights).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if tb.writer:
            tb.writer.add_scalar("loss/loss", loss, step)
            tb.writer.add_scalar("loss/energy_loss", loss, step)
            tb.writer.add_scalar("param/mean_deformation", deformations.mean(), step)

        if args.decay_lrate:
            new_lrate = args.lrate * args.lrate_decay_rate ** (step / args.lrate_decay_steps)
            for param_group in optimizer.param_groups:
                param_group["lr"] = new_lrate

        if not step % args.i_summary:
            logger.info(f"Iteration: {step}, loss: {loss}, mean_deformation: {deformations.mean()}")

        if not step % args.i_weights:
            torch.save(
                {
                    "global_step": step,
                    "siren_state_dict": deformation_field.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                os.path.join(weights_dir, f"{step:06d}.tar"),
            )

        if not step % args.i_test:
            evaluate(args, deformation_field, rest_surface, meshes_dir, step)

    if args.run_render and final_step % args.i_test:
        evaluate(args, deformation_field, rest_surface, meshes_dir, final_step)

    if tb.writer:
        tb.writer.flush()
        tb.writer.close()

    run_postprocessing(args, log_dir, logger)


if __name__ == "__main__":
    train()
