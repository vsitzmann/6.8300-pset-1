import torch
from jaxtyping import install_import_hook

# Add runtime type checking to all imports.
with install_import_hook(("src",), ("beartype", "beartype")):
    from src.geometry import (
        homogenize_points,
        homogenize_vectors,
        project,
        transform_cam2world,
        transform_rigid,
        transform_world2cam,
    )


def f32(x):
    return torch.tensor(x, dtype=torch.float32)


def test_homogenize_points():
    assert torch.allclose(
        homogenize_points(f32([1, 2, 3])),
        f32([1, 2, 3, 1]),
    )


def test_homogenize_vectors():
    assert torch.allclose(
        homogenize_vectors(f32([1, 2, 3])),
        f32([1, 2, 3, 0]),
    )


def test_transform_rigid():
    t = torch.eye(4, dtype=torch.float32)
    t[0, 3] = 1
    assert torch.allclose(
        transform_rigid(f32([1, 2, 3, 1]), t),
        f32([2, 2, 3, 1]),
    )


def test_transform_world2cam():
    t = torch.eye(4, dtype=torch.float32)
    t[1, 3] = 0.5
    assert torch.allclose(
        transform_world2cam(f32([0, 0, 0, 1]), t),
        f32([0, -0.5, 0, 1]),
    )


def test_transform_cam2world():
    t = torch.eye(4, dtype=torch.float32)
    t[1, 3] = 5.0
    assert torch.allclose(
        transform_cam2world(f32([0, 0, 0, 1]), t),
        f32([0, 5.0, 0, 1]),
    )


def test_project():
    xyz = f32([1, 1, 2, 1])
    intrinsics = f32([[0.25, 0, 0.5], [0, 0.25, 0.5], [0, 0, 1]])
    assert torch.allclose(
        project(xyz, intrinsics),
        f32([0.625, 0.625]),
    )
