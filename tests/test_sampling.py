def test_grid_sampler_returns_expected_shape():
    pytest = __import__("pytest")
    torch = pytest.importorskip("torch")

    from garment_tdc.sampling import DomainBounds, GridSampler

    sampler = GridSampler(16, DomainBounds(2.0, 3.0))
    rest_points = sampler[0]

    assert rest_points.shape == (16, 2)
    assert rest_points.requires_grad
    assert torch.all(rest_points[:, 0] >= 0)
    assert torch.all(rest_points[:, 0] <= 2.0)
    assert torch.all(rest_points[:, 1] >= 0)
    assert torch.all(rest_points[:, 1] <= 3.0)
