from types import SimpleNamespace


def test_linear_shell_energy_zero_deformation_smoke():
    pytest = __import__("pytest")
    torch = pytest.importorskip("torch")

    from garment_tdc.energy import LinearMaterial, ShellEnergy

    points = torch.tensor(
        [[[0.0, 0.0, 0.0],
          [1.0, 0.0, 0.0],
          [0.0, 1.0, 0.0],
          [1.0, 1.0, 0.0]]],
        requires_grad=True,
    )
    normals = torch.tensor([[[0.0, 0.0, 1.0]]]).expand(1, 4, 3)
    identity = torch.eye(3).view(1, 1, 3, 3)
    projector = identity - normals.unsqueeze(-1) @ normals.unsqueeze(-2)

    ref_geometry = SimpleNamespace(
        points=points,
        normals=normals,
        projector=projector,
        weingarten=torch.zeros(1, 4, 3, 3),
        area_element=torch.ones(1, 4),
        n_spatial_samples=4,
    )
    args = SimpleNamespace(
        mass_area_density=1.0,
        thickness=0.01,
        poissons_ratio=0.3,
        youngs_modulus=1000.0,
    )

    material = LinearMaterial(args, ref_geometry)
    energy = ShellEnergy(ref_geometry, material, [0.0, 0.0, 0.0], 1)
    deformations = points * 0.0

    value = energy(deformations, 0)

    assert value.shape == (1, 4)
    assert torch.isfinite(value).all()
    assert torch.allclose(value, torch.zeros_like(value), atol=1e-8)
