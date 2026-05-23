import argparse
import os
from glob import glob
import xml.etree.ElementTree as ET

from tqdm import tqdm


def load_render_dependencies():
    try:
        import imageio.v2 as imageio
        import numpy as np
        import pyvista as pv
    except ImportError as exc:
        package = exc.name or "required rendering package"
        raise SystemExit(
            f"Missing dependency '{package}'. Install the rendering dependencies with "
            "`pip install -e .[geometry]` or `uv pip install -e .[geometry]` inside your active environment."
        ) from exc
    return imageio, np, pv


CAMERA_1_XML = '''<!DOCTYPE ViewState>
<project>
 <VCGCamera TranslationVector="0 0 5 1" 
            RotationMatrix="1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"
            ViewportPx="1399 1010"
            PixelSizeMm="0.0369161 0.0369161"
            CenterPx="699 505"
            FocalMm="50.0"/>
 <ViewSettings FarPlane="100.0" NearPlane="0.1"/>
</project>'''

CAMERA_2_XML = '''<!DOCTYPE ViewState>
<project>
 <VCGCamera TranslationVector="5 0 0 1" 
            RotationMatrix="0 0 1 0 0 1 0 0 -1 0 0 0 0 0 0 1"
            ViewportPx="1399 1010" 
            PixelSizeMm="0.0369161 0.0369161"
            CenterPx="699 505"
            FocalMm="50.0"/>
 <ViewSettings FarPlane="100.0" NearPlane="0.1"/>
</project>'''

def meshlab_to_dict(xml_str, meshes):
    """Convert Meshlab XML to PyVista camera parameters with robust auto-framing."""
    _, np, _ = load_render_dependencies()
    try:
        root = ET.fromstring(xml_str)
        vcg_camera = root.find('.//VCGCamera')
        
        translation = list(map(float, vcg_camera.get('TranslationVector').split()))
        initial_position = np.array(translation[:3])
        
        rotation = list(map(float, vcg_camera.get('RotationMatrix').split()))
        rotation_matrix = np.array(rotation).reshape(4, 4)[:3, :3]

        view_up = rotation_matrix[:, 1]

        bounds = np.array([mesh.bounds for mesh in meshes])
        scene_min = bounds[:, 0:5:2].min(axis=0)
        scene_max = bounds[:, 1:6:2].max(axis=0)
        scene_center = (scene_min + scene_max) / 2.0

        focal_point = scene_center

        scene_size = np.linalg.norm(scene_max - scene_min)
        if np.isclose(scene_size, 0):
            scene_size = 1.0

        direction = initial_position - focal_point
        norm_direction = np.linalg.norm(direction)
        
        if np.isclose(norm_direction, 0):
            unit_direction = np.array([0.0, 0.0, 1.0])
        else:
            unit_direction = direction / norm_direction
        
        distance = scene_size * 1.5 
        position = focal_point + unit_direction * distance

        final_distance_to_focal = np.linalg.norm(position - focal_point)
        near_clip = max(0.01, final_distance_to_focal - scene_size * 2.0)
        far_clip = final_distance_to_focal + scene_size * 4.0
        
        if near_clip >= far_clip:
            near_clip = far_clip / 1000.0

        camera_params = {
            'position': position,
            'focal_point': focal_point,
            'view_up': view_up,
            'clipping_range': (near_clip, far_clip)
        }
        
        return camera_params, scene_center, scene_size
        
    except Exception as e:
        print(f"Error setting up camera from XML: {e}")
        return None, None, None


def get_gif_output(args, mesh_body_path, mesh_cloth_paths, view, output_file):
    """Generate a GIF from a sequence of meshes with a reliable, stable camera."""
    imageio, np, pv = load_render_dependencies()
    frame_size = (args.width, args.height)

    plotter = pv.Plotter(off_screen=True, window_size=frame_size)
    plotter.background_color = "#c2b3a7"
    plotter.enable_shadows()
    cloth_colour = "#3087af"
    mesh_colour = "#525048"

    mesh_body = None
    if mesh_body_path and os.path.exists(mesh_body_path):
        try:
            mesh_body = pv.read(mesh_body_path)
        except Exception as e:
            print(f"Error loading body mesh: {e}")

    all_meshes_for_bounds = []
    if mesh_body:
        all_meshes_for_bounds.append(mesh_body)
    
    for path in tqdm(mesh_cloth_paths, desc="Reading mesh bounds"):
        try:
            all_meshes_for_bounds.append(pv.read(path))
        except Exception as e:
            print(f"Warning: Could not read mesh for bounds calculation: {path}, {e}")
            continue

    if not all_meshes_for_bounds:
        raise RuntimeError("No valid meshes found. Cannot generate GIF.")

    scene_size = None
    if view in (1, 2):
        xml_str = CAMERA_1_XML if view == 1 else CAMERA_2_XML
        cam_params, _, scene_size = meshlab_to_dict(xml_str, all_meshes_for_bounds)
    else:
        cam_params = None

    if cam_params:
        plotter.camera.position = cam_params['position']
        plotter.camera.focal_point = cam_params['focal_point']
        plotter.camera.up = cam_params['view_up']
        plotter.camera.clipping_range = cam_params['clipping_range']
    else:
        temp_plotter = pv.Plotter(off_screen=True)
        for mesh in all_meshes_for_bounds:
            temp_plotter.add_mesh(mesh)
        temp_plotter.reset_camera()
        
        all_bounds = np.array([m.bounds for m in all_meshes_for_bounds])
        scene_min = all_bounds[:, 0:5:2].min(axis=0)
        scene_max = all_bounds[:, 1:6:2].max(axis=0)
        scene_size = np.linalg.norm(scene_max - scene_min)
        
        plotter.camera.position = temp_plotter.camera.position
        plotter.camera.focal_point = temp_plotter.camera.focal_point
        plotter.camera.up = temp_plotter.camera.up
        plotter.camera.clipping_range = temp_plotter.camera.clipping_range
        temp_plotter.close()

    key_light = pv.Light(
        position=(scene_size, scene_size, scene_size),
        intensity=1.0,
        color='white'
    )
    fill_light = pv.Light(
        position=(-scene_size, -scene_size, scene_size * 0.5),
        intensity=0.8,
        color='white'
    )
    headlight = pv.Light(light_type='headlight', intensity=0.8)

    frames_per_mesh = max(1, int(args.fps * args.seconds_per_mesh))
    
    last_valid_img = None
    pbar = tqdm(mesh_cloth_paths, desc="Rendering frames")

    with imageio.get_writer(output_file, mode="I", duration=1.0 / args.fps, loop=0) as gif:
        for mesh_path in pbar:
            try:
                mesh_cloth = pv.read(mesh_path)
                
                plotter.clear()
                
                plotter.add_light(key_light)
                plotter.add_light(fill_light)
                plotter.add_light(headlight)

                plotter.add_mesh(
                    mesh_cloth, 
                    color=cloth_colour, 
                    smooth_shading=True, 
                    specular=0.3,
                    diffuse=0.7
                )
                if mesh_body:
                    plotter.add_mesh(
                        mesh_body, 
                        color=mesh_colour, 
                        opacity=args.opacity, 
                        smooth_shading=True, 
                        specular=0.1,
                        diffuse=0.7
                    )
                
                img = plotter.screenshot()
                
                if img is None:
                    print(f"Warning: Screenshot failed for mesh {mesh_path}. Skipping frame.")
                    continue

                for _ in range(frames_per_mesh):
                    gif.append_data(img)
                
                last_valid_img = img
                    
            except Exception as e:
                print(f"Error processing mesh {mesh_path}: {e}. Skipping frame.")
                continue

        if last_valid_img is not None:
            for _ in tqdm(range(args.frame_delay), desc="Adding final hold frames"):
                gif.append_data(last_valid_img)

    plotter.close()
    print(f"\nGIF successfully saved to {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog="MeshVid",
        description="Convert a series of meshes into GIFs with reliable, stable camera views."
    )
    parser.add_argument("--expt_name", required=True, type=str, 
                       help='Experiment name containing the mesh sequence folder inside "logs/"')
    parser.add_argument("--logs_dir", default="logs", type=str,
                       help='Root directory containing experiment logs.')
    parser.add_argument("--no_body", action='store_true', 
                       help='Exclude the body mesh from the rendering.')
    parser.add_argument("--output_dir", type=str, default=None,
                       help='Output directory for GIF files.')
    parser.add_argument("--seconds_per_mesh", type=float, default=0.1,
                       help="Seconds to display each mesh frame.")
    parser.add_argument("--frame_delay", type=int, default=30,
                       help="Number of additional frames to hold on the final mesh.")
    parser.add_argument("--opacity", type=float, default=0.7,
                       help='Opacity of the body mesh.')
    parser.add_argument('--fps', type=int, default=30,
                       help='GIF frames per second.')
    parser.add_argument("--width", type=int, default=512,
                       help='Frame width in pixels.')
    parser.add_argument("--height", type=int, default=512,
                       help='Frame height in pixels.')
    parser.add_argument("--views", type=int, choices=[0, 1, 2], nargs="+", default=[1, 2],
                       help="Camera views. 0: auto-frame, 1: front view, 2: side view.")

    args = parser.parse_args()
    
    expt_folder = os.path.join(args.logs_dir, args.expt_name)
    if not os.path.exists(expt_folder):
        raise FileNotFoundError(f"Experiment folder not found: {expt_folder}")
    
    mesh_cloth_paths = sorted(
        [f for f in glob(f"{expt_folder}/meshes/*/*.obj") if "body_" not in os.path.basename(f)],
        key=lambda x: int(os.path.basename(os.path.dirname(x)))
    )
    
    if not mesh_cloth_paths:
        raise FileNotFoundError(f"No cloth meshes found in the subdirectories of {expt_folder}/meshes/")
    
    mesh_body_path = None
    if not args.no_body:
        body_candidates = glob(f"{expt_folder}/meshes/*/body_*.obj")
        if body_candidates:
            mesh_body_path = body_candidates[0]
        else:
            print("Warning: --no_body was not specified, but no body mesh was found.")
    
    output_dir = args.output_dir or expt_folder
    os.makedirs(output_dir, exist_ok=True)

    try:
        for view in args.views:
            output_file = os.path.join(output_dir, f"{args.expt_name}_view{view}.gif")
            get_gif_output(args, mesh_body_path, mesh_cloth_paths, view, output_file)
    except (RuntimeError, FileNotFoundError) as e:
        print(f"\nAn error occurred: {e}")
