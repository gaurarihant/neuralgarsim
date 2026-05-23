def test_package_imports():
    import garment_tdc
    assert garment_tdc is not None


def test_core_package_does_not_import_removed_collision_modules():
    import sys
    import garment_tdc

    assert garment_tdc is not None
    forbidden = ("collision", "run_sdf", "body_sdf", "smpl")
    imported = {name.lower() for name in sys.modules}
    assert not any(token in module_name for token in forbidden for module_name in imported)
