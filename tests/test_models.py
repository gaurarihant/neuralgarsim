def test_siren_deformation_field_forward_shape():
    pytest = __import__("pytest")
    torch = pytest.importorskip("torch")

    from garment_tdc.boundary_conditions import BoundaryCondition
    from garment_tdc.models import SirenDeformationField

    boundary = BoundaryCondition("rectangle_xy", "free_fall")
    model = SirenDeformationField(
        boundary,
        in_features=3,
        hidden_features=8,
        hidden_layers=1,
        out_features=3,
    )

    rest_points = torch.zeros(1, 5, 3, requires_grad=True)
    deformation = model(rest_points)

    assert deformation.shape == (1, 5, 3)


def test_mesh_vertices_moved_uses_configured_direction():
    pytest = __import__("pytest")
    torch = pytest.importorskip("torch")

    from garment_tdc.boundary_conditions import BoundaryCondition

    boundary_points = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    points = boundary_points.view(1, 2, 3)
    deformations = torch.zeros_like(points)

    away = BoundaryCondition(
        "mesh",
        "mesh_vertices_moved",
        boundary_points=boundary_points,
        moved_boundary_mode="away",
        moved_boundary_displacement=0.25,
    )
    away_result = away.dirichlet_condition(deformations.clone(), points)

    towards = BoundaryCondition(
        "mesh",
        "mesh_vertices_moved",
        boundary_points=boundary_points,
        moved_boundary_mode="towards",
        moved_boundary_displacement=0.25,
    )
    towards_result = towards.dirichlet_condition(deformations.clone(), points)

    assert torch.allclose(away_result[0, 0], torch.tensor([-0.25, 0.0, 0.0]))
    assert torch.allclose(away_result[0, 1], torch.tensor([0.25, 0.0, 0.0]))
    assert torch.allclose(towards_result[0, 0], torch.tensor([0.25, 0.0, 0.0]))
    assert torch.allclose(towards_result[0, 1], torch.tensor([-0.25, 0.0, 0.0]))
