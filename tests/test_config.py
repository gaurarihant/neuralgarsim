def test_bracketed_list_config_values_parse():
    pytest = __import__("pytest")
    pytest.importorskip("configargparse")
    pytest.importorskip("torch")

    from garment_tdc.config import get_config_parser

    args = get_config_parser().parse_args([
        "-c",
        "configs/garments/smoke_test.ini",
        "--reference_boundary_vertices",
        "[1, 2, 3]",
        "--moved_boundary_mode",
        "towards",
        "--moved_boundary_displacement",
        "0.125",
        "--debug",
        "False",
        "--no_reload",
        "True",
    ])

    assert args.gravity_acceleration == [0.0, -9.8, 0.0]
    assert args.reference_boundary_vertices == [1, 2, 3]
    assert args.moved_boundary_mode == "towards"
    assert args.moved_boundary_displacement == 0.125
    assert args.debug is False
    assert args.no_reload is True
