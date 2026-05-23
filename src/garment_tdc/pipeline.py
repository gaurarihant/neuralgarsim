import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_repo_path(path):
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def _run_pipeline_command(command, logger, stage_name):
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    if result.stdout:
        logger.info(result.stdout.strip())
    if result.stderr:
        logger.error(result.stderr.strip())
    if result.returncode != 0:
        raise RuntimeError(
            f"{stage_name} failed with exit code {result.returncode}. "
            "See the log above for the underlying error."
        )


def run_mesh_preprocessing(args, logger):
    """Run optional OBJ normalisation and point the simulation at the result."""
    source = Path(args.reference_geometry_source)
    output_dir = Path(args.preprocess_output_dir)
    source_abs = _resolve_repo_path(source)

    if not source_abs.exists():
        raise FileNotFoundError(
            f"Preprocessing input mesh not found: {source}. "
            "Set reference_geometry_source to an existing OBJ, or set preprocess_mesh = False "
            "if the mesh is already prepared."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    output_source = output_dir / f"{args.preprocess_prefix}{source.name}"
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "preprocess" / "normalize_mesh_asset.py"),
        "--input",
        str(source),
        "--output-dir",
        str(output_dir),
        "--prefix",
        args.preprocess_prefix,
        "--centre-method",
        args.preprocess_centre_method,
        "--target-max-abs",
        str(args.preprocess_target_max_abs),
        "--axis-transform",
        args.preprocess_axis_transform,
    ]

    logger.info("Running mesh preprocessing: %s", " ".join(command))
    _run_pipeline_command(command, logger, "Mesh preprocessing")

    args.reference_geometry_source = str(output_source)
    if not _resolve_repo_path(args.reference_geometry_source).exists():
        raise FileNotFoundError(f"Preprocessing finished but did not create {args.reference_geometry_source}")
    logger.info("Using preprocessed mesh: %s", args.reference_geometry_source)
    return args.reference_geometry_source


def run_rendering(args, logger):
    output_dir = args.render_output_dir or os.path.join(args.logging_dir, args.expt_name)
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "render" / "render_mesh_sequence.py"),
        "--expt_name",
        args.expt_name,
        "--logs_dir",
        args.logging_dir,
        "--output_dir",
        output_dir,
        "--seconds_per_mesh",
        str(args.render_seconds_per_mesh),
        "--frame_delay",
        str(args.render_frame_delay),
        "--opacity",
        str(args.render_opacity),
        "--fps",
        str(args.render_fps),
        "--width",
        str(args.render_width),
        "--height",
        str(args.render_height),
        "--views",
        *[str(view) for view in args.render_views],
    ]
    if args.render_no_body:
        command.append("--no_body")

    logger.info("Running rendering: %s", " ".join(command))
    _run_pipeline_command(command, logger, "Rendering")
    return output_dir


def run_postprocessing(args, log_dir, logger):
    if args.run_render:
        run_rendering(args, logger)
