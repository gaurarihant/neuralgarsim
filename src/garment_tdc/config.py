import math
import configparser
from pathlib import Path
import configargparse
import torch
import ast


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


MATERIAL_PRESETS = {"linear_1", "cotton", "silk", "canvas"}
MATERIAL_CONFIG_KEYS = {
    "material_type": str,
    "mass_area_density": float,
    "thickness": float,
    "youngs_modulus": float,
    "poissons_ratio": float,
    "a11": float,
    "a12": float,
    "a22": float,
    "G12": float,
    "d": lambda value: parse_list(value, int),
    "mu1": lambda value: parse_list(value, float),
    "mu2": lambda value: parse_list(value, float),
    "mu3": lambda value: parse_list(value, float),
    "mu4": lambda value: parse_list(value, float),
    "alpha1": lambda value: parse_list(value, float),
    "alpha2": lambda value: parse_list(value, float),
    "alpha3": lambda value: parse_list(value, float),
    "alpha4": lambda value: parse_list(value, float),
    "E_11_min": float,
    "E_11_max": float,
    "E_22_min": float,
    "E_22_max": float,
    "E_12_max": float,
}


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    value = value.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise configargparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}.")


def parse_list(value, item_type):
    """Parse configargparse list values from either INI or CLI syntax."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if all(isinstance(item, str) for item in value):
            return parse_list(" ".join(value), item_type)
        parsed = []
        for item in value:
            parsed.extend(parse_list(item, item_type))
        return parsed
    if isinstance(value, item_type):
        return [value]

    text = str(value).strip()
    if not text:
        return []

    try:
        literal = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        literal = [part for part in text.replace(",", " ").split() if part]

    if isinstance(literal, (list, tuple)):
        return [item_type(item) for item in literal]
    return [item_type(literal)]


class GarmentConfigParser(configargparse.ArgumentParser):
    LIST_ARGUMENT_TYPES = {
        "gravity_acceleration": float,
        "reference_boundary_vertices": int,
        "d": int,
        "mu1": float,
        "mu2": float,
        "mu3": float,
        "mu4": float,
        "alpha1": float,
        "alpha2": float,
        "alpha3": float,
        "alpha4": float,
        "render_views": int,
    }

    def parse_args(self, *args, **kwargs):
        parsed = super().parse_args(*args, **kwargs)
        apply_material_config(parsed)
        for name, item_type in self.LIST_ARGUMENT_TYPES.items():
            setattr(parsed, name, parse_list(getattr(parsed, name), item_type))
        return parsed


def material_path_from_preset(name):
    return str(Path("material") / f"{name}.ini")


def apply_material_config(args):
    requested_material = args.material_type

    if args.material_filepath is None:
        if requested_material in MATERIAL_PRESETS:
            args.material_name = requested_material
            args.material_filepath = material_path_from_preset(requested_material)
        else:
            args.material_name = "linear_1"
            args.material_filepath = material_path_from_preset("linear_1")
    else:
        args.material_name = Path(args.material_filepath).stem

    material_parser = configparser.ConfigParser()
    material_parser.optionxform = str
    read_files = material_parser.read(args.material_filepath)
    if not read_files:
        raise FileNotFoundError(f"Material config not found: {args.material_filepath}")

    for section in material_parser.sections() + [material_parser.default_section]:
        if section != material_parser.default_section and not material_parser.has_section(section):
            continue
        items = material_parser.defaults().items() if section == material_parser.default_section else material_parser.items(section)
        for key, value in items:
            if key in MATERIAL_CONFIG_KEYS:
                setattr(args, key, MATERIAL_CONFIG_KEYS[key](value))

    if requested_material not in (None, *MATERIAL_PRESETS):
        args.material_type = requested_material


def get_config_parser():
    parser = GarmentConfigParser()
    parser.add('-c', '--config_filepath', required=True, is_config_file=True, help='config file path')
    parser.add_argument('-n', '--expt_name', type=str, required=True, help='experiment name; this will also be the name of subdirectory in logging_dir')

    # simulation parameters
    parser.add_argument('--reference_geometry_name', type=str, default='rectangle_xy', help='name of the reference geometry; can be rectangle_xy, rectangle_xz, cylinder, cone or mesh')
    parser.add_argument('--reference_geometry_type', type=str, default='udf', choices=['udf', 'mesh', 'pcd', 'gaussians'], help='type of the reference geometry; only udf fits a reference neural field')
    parser.add_argument('--domain_u_max', '--xi__1_max', dest='domain_u_max', type=float, default=2 * math.pi, help='upper analytical-surface domain bound along the first coordinate')
    parser.add_argument('--domain_v_max', '--xi__2_max', dest='domain_v_max', type=float, default=1, help='upper analytical-surface domain bound along the second coordinate')
    parser.add_argument('--boundary_condition_name', type=str, default='top_left_fixed', help='boundary condition name; mesh garments support mesh_vertices, mesh_vertices_moved, and free_fall')
    parser.add_argument('--moved_boundary_mode', type=str, choices=['away', 'towards'], default='away', help='direction for mesh_vertices_moved handles')
    parser.add_argument('--moved_boundary_displacement', type=float, default=0.05, help='distance used by mesh_vertices_moved handles')
    parser.add_argument('--gravity_acceleration', nargs='*', default=[0, -9.8, 0], help='acceleration due to gravity')

    # model params
    parser.add_argument('--hidden_layers', type=int, default=5, help='number of hidden layers in the deformation field')
    parser.add_argument('--siren_omega', type=float, default=30., help='first and hidden omega value for the deformation field')
    parser.add_argument('--hidden_features', type=int, default=512, help='number of hidden features in the deformation field')
    parser.add_argument('--use_udf', action='store_true', help='sample from the fitted UDF when reference_geometry_type is udf')

    # additional parameters if the reference geometry is a mesh
    parser.add_argument('--reference_geometry_source', type=str, default=None, help='OBJ file of the reference garment. Required when reference_geometry_name is mesh')
    parser.add_argument('--reference_mlp_n_iterations', type=int, default=3000, help='number of iterations for fitting the reference geometry MLP when reference_geometry_type is udf')
    parser.add_argument('--reference_mlp_lrate', type=float, default=5e-6, help='learning rate for the reference geometry MLP when reference_geometry_type is udf')
    parser.add_argument('--reference_boundary_vertices', nargs='*', help='vertices for boundary condition on the reference geometry')
    parser.add_argument('--use_pretrained_ref', type=str_to_bool, nargs='?', const=True, default=True, help='use pretrained weights for reference MLP to save on time')
    parser.add_argument('--rep_type', type=str, default="DUDF", choices=["DUDF", "SDF", "UDF"], help='type of garment representation')

    # material parameters
    parser.add('-m', '--material_filepath', default=None, help='material config path; if omitted, material_type can be a preset such as cotton, silk, canvas, or linear_1')
    parser.add_argument('--material_type', type=str, help='material model type (linear/nonlinear) or a material preset name')
    parser.add_argument('--StVK', action='store_true', help='whether to use the St.Venant-Kirchhoff simplification of the Clyde material model')
    
    parser.add_argument('--mass_area_density', type=float, default=0.144, help='mass area density in kg/m^2 ')
    parser.add_argument('--thickness', type=float, default=0.0012, help='thickness in meters')
    
    # material parameters for a linear material
    parser.add_argument('--youngs_modulus', type=float, default=5000, help='Young\'s modulus')
    parser.add_argument('--poissons_ratio', type=float, default=0.25, help='Poisson\'s ratio')

    # material parameters for a nonlinear material [Clyde et al., 2017]
    parser.add_argument('--a11', type=float, help='a11')
    parser.add_argument('--a12', type=float, help='a12')
    parser.add_argument('--a22', type=float, help='a22')
    parser.add_argument('--G12', type=float, help='G12')
        
    parser.add_argument('--d', nargs='*', help='degree')
    parser.add_argument('--mu1', nargs='*', help='mu1')
    parser.add_argument('--mu2', nargs='*', help='mu2')
    parser.add_argument('--mu3', nargs='*', help='mu3')
    parser.add_argument('--mu4', nargs='*', help='mu4')
    parser.add_argument('--alpha1', nargs='*', help='alpha1')
    parser.add_argument('--alpha2', nargs='*', help='alpha2')
    parser.add_argument('--alpha3', nargs='*', help='alpha3')
    parser.add_argument('--alpha4', nargs='*', help='alpha4')
    
    parser.add_argument('--E_11_min', type=float, help='E_11_min')
    parser.add_argument('--E_11_max', type=float, help='E_11_max')
    parser.add_argument('--E_22_min', type=float, help='E_22_min')
    parser.add_argument('--E_22_max', type=float, help='E_22_max')
    parser.add_argument('--E_12_max', type=float, help='E_12_max')
    
    # training options
    parser.add_argument('--train_n_spatial_samples', type=int, default=400, help='N_omega, number of samples used for training; when the reference geometry is an analytical surface, the number of spatial grid samples along each analytical coordinate is square_root(N_omega)')
    parser.add_argument('--lrate', type=float, default=5e-6, help='learning rate')
    parser.add_argument('--decay_lrate', type=str_to_bool, nargs='?', const=True, default=True, help='whether to decay learning rate')
    parser.add_argument('--lrate_decay_steps', type=int, default=5000, help='learning rate decay steps')
    parser.add_argument('--lrate_decay_rate', type=float, default=0.1, help='learning rate decay rate')    
    parser.add_argument('--n_iterations', type=int, default=5001, help='total number of training iterations')
    parser.add_argument("--init_deformation_ckpt", "--init_ndf_ckpt", dest="init_deformation_ckpt", type=str, default=None, help="checkpoint used to warm-start the deformation field")
    parser.add_argument("--edit_lrate", type=float, default=1e-5)

    # optional preprocessing
    parser.add_argument('--preprocess_mesh', type=str_to_bool, nargs='?', const=True, default=False, help='normalise the input OBJ before simulation')
    parser.add_argument('--preprocess_output_dir', type=str, default='data/garments/processed', help='directory for preprocessed meshes')
    parser.add_argument('--preprocess_prefix', type=str, default='normalised_', help='prefix for preprocessed mesh filenames')
    parser.add_argument('--preprocess_centre_method', type=str, choices=['mean', 'bounds', 'centre_mass'], default='mean', help='centering rule for mesh preprocessing')
    parser.add_argument('--preprocess_target_max_abs', type=float, default=1.0, help='target maximum absolute coordinate after preprocessing')
    parser.add_argument('--preprocess_axis_transform', type=str, choices=['none', 'y_to_z', 'y_to_minus_z'], default='none', help='optional right-handed dataset-specific axis transform during preprocessing')

    # optional post-processing
    parser.add_argument('--run_render', type=str_to_bool, nargs='?', const=True, default=False, help='render the saved mesh sequence after training')
    parser.add_argument('--render_output_dir', type=str, default=None, help='directory for GIF render outputs; defaults to logs/<expt_name>')
    parser.add_argument('--render_seconds_per_mesh', type=float, default=0.1, help='seconds to display each saved mesh frame')
    parser.add_argument('--render_frame_delay', type=int, default=30, help='additional frames to hold on the final mesh')
    parser.add_argument('--render_opacity', type=float, default=0.7, help='body mesh opacity when a body mesh is present')
    parser.add_argument('--render_fps', type=int, default=30, help='rendered GIF frames per second')
    parser.add_argument('--render_width', type=int, default=512, help='rendered GIF width in pixels')
    parser.add_argument('--render_height', type=int, default=512, help='rendered GIF height in pixels')
    parser.add_argument('--render_views', nargs='*', default=[1, 2], help='camera views to render as GIFs: 0 auto-frame, 1 front, 2 side')
    parser.add_argument('--render_no_body', type=str_to_bool, nargs='?', const=True, default=True, help='skip body mesh rendering')

    # logging/saving options
    parser.add_argument('--logging_dir', type=str, default='logs', help='root directory for logging')
    parser.add_argument("--i_weights", type=int, default=400, help='frequency of saving deformation-field weights as checkpoints')
    parser.add_argument("--i_summary", type=int, default=100, help='frequency of logging losses')
    parser.add_argument("--i_test", type=int, default=100, help='frequency of evaluating the deformation field and saving simulated meshes during training')
    
    # debug options
    parser.add_argument('--debug', type=str_to_bool, nargs='?', const=True, default=True, help='whether to run in debug mode; this will log the reference geometric quantities (e.g. metric and curvature tensor), the strains, and the simulated states to TensorBoard')
    parser.add_argument('--i_debug', type=int, default=200, help='frequency of Tensordboard logging')
    
    # reload options
    parser.add_argument('--no_reload', type=str_to_bool, nargs='?', const=True, default=False, help='do not resume training from checkpoint, rather train from scratch')
    parser.add_argument("--i_ckpt", type=int, help='weight checkpoint to reload for resuming training or performing evaluation; if None, the latest checkpoint is used')
    parser.add_argument('--test_only', type=str_to_bool, nargs='?', const=True, default=False, help='evaluate the deformation field from i_ckpt and save simulated meshes; do not resume training')
    
    # testing options
    parser.add_argument('--test_n_spatial_samples', type=int, default=400, help='N_omega, the number of samples used for evaluation when reference geometry is an analytical surface; if the reference geometry is instead a mesh, this argument is ignored, and the samples (vertices and faces) used for evaluation will match that of template mesh') 

    return parser
