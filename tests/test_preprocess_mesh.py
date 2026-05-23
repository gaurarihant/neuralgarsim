def test_preprocess_mesh_normalises_without_axis_swap(tmp_path):
    pytest = __import__("pytest")
    np = pytest.importorskip("numpy")

    from scripts.preprocess.normalize_mesh_asset import process_mesh

    src = tmp_path / "mesh.obj"
    dst = tmp_path / "out.obj"
    src.write_text(
        "\n".join([
            "# preserve comments",
            "v 1.0 2.0 3.0",
            "v 3.0 2.0 3.0",
            "v 1.0 4.0 3.0",
            "f 1 2 3",
            "",
        ]),
        encoding="utf-8",
    )

    process_mesh(
        src,
        dst,
        centre_method="bounds",
        target_max_abs=1.0,
        axis_transform="none",
    )

    lines = dst.read_text(encoding="utf-8").splitlines()
    vertex_lines = [line for line in lines if line.startswith("v ")]
    vertices = np.asarray([[float(value) for value in line.split()[1:4]] for line in vertex_lines])

    assert lines[0] == "# preserve comments"
    assert lines[-1] == "f 1 2 3"
    assert len(vertex_lines) == 3
    assert np.allclose(0.5 * (vertices.min(axis=0) + vertices.max(axis=0)), 0.0)
    assert np.isclose(np.abs(vertices).max(), 1.0)
    assert np.allclose(vertices[:, 2], 0.0)


def test_axis_transforms_are_right_handed():
    pytest = __import__("pytest")
    np = pytest.importorskip("numpy")

    from scripts.preprocess.normalize_mesh_asset import axis_transform_matrix

    assert np.isclose(np.linalg.det(axis_transform_matrix("none")), 1.0)
    assert np.isclose(np.linalg.det(axis_transform_matrix("y_to_z")), 1.0)
    assert np.isclose(np.linalg.det(axis_transform_matrix("y_to_minus_z")), 1.0)
