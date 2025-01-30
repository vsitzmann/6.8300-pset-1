import urllib.request
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import rearrange, repeat
from jaxtyping import Float, Int, UInt8
from PIL import Image
from scipy.spatial.transform import Rotation as R
from torch import Tensor
from trimesh.exchange.obj import load_obj

FloatImage = Union[
    Float[Tensor, "height width"],
    Float[Tensor, "channel height width"],
    Float[Tensor, "batch channel height width"],
]


def prep_image(image: FloatImage) -> UInt8[np.ndarray, "height width channel"]:
    # Handle batched images.
    if image.ndim == 4:
        image = rearrange(image, "b c h w -> c h (b w)")

    # Handle single-channel images.
    if image.ndim == 2:
        image = rearrange(image, "h w -> () h w")

    # Ensure that there are 3 or 4 channels.
    channel, _, _ = image.shape
    if channel == 1:
        image = repeat(image, "() h w -> c h w", c=3)
    assert image.shape[0] in (3, 4)

    image = (image.detach().clip(min=0, max=1) * 255).type(torch.uint8)
    return rearrange(image, "c h w -> h w c").cpu().numpy()


def save_image(
    image: FloatImage,
    path: Union[Path, str],
) -> None:
    """Save an image. Assumed to be in range 0-1."""

    # Create the parent directory if it doesn't already exist.
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)

    # Save the image.
    Image.fromarray(prep_image(image)).save(path)


def download_file(url: str, path: Path) -> None:
    """Download a file from the specified URL."""
    if path.exists():
        return
    path.parent.mkdir(exist_ok=True, parents=True)
    urllib.request.urlretrieve(url, path)


def load_mesh(
    path: Path, device: torch.device = torch.device("cpu")
) -> tuple[Float[Tensor, "vertex 3"], Int[Tensor, "face 3"]]:
    """Load a mesh."""
    with path.open("r") as f:
        mesh = load_obj(f)
    vertices = torch.tensor(mesh["vertices"], dtype=torch.float32, device=device)
    faces = torch.tensor(mesh["faces"], dtype=torch.int64, device=device)
    return vertices, faces


def get_bunny(
    bunny_url: str = "https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/stanford-bunny.obj",
    bunny_path: Path = Path("data/stanford_bunny.obj"),
    device: torch.device = torch.device("cpu"),
) -> tuple[Float[Tensor, "vertex 3"], Int[Tensor, "face 3"]]:
    download_file(bunny_url, bunny_path)
    vertices, faces = load_mesh(bunny_path, device=device)

    # Center and rescale the bunny.
    maxima, _ = vertices.max(dim=0, keepdim=True)
    minima, _ = vertices.min(dim=0, keepdim=True)
    centroid = 0.5 * (maxima + minima)
    vertices -= centroid
    vertices /= (maxima - minima).max()

    return vertices, faces


def generate_spin(
    num_steps: int,
    elevation: float,
    radius: float,
    device: torch.device = torch.device("cpu"),
) -> Float[Tensor, "batch 4 4"]:
    # Translate back along the camera's look vector.
    tf_translation = torch.eye(4, dtype=torch.float32, device=device)
    tf_translation[2, 3] = -radius
    tf_translation[:2, :2] *= -1  # Use +Y as world up instead of -Y.

    # Generate the transformation for the azimuth.
    t = np.linspace(0, 1, num_steps, endpoint=False)
    azimuth = [
        R.from_rotvec(np.array([0, x * 2 * np.pi, 0], dtype=np.float32)).as_matrix()
        for x in t
    ]
    azimuth = torch.tensor(np.array(azimuth), dtype=torch.float32, device=device)
    tf_azimuth = torch.eye(4, dtype=torch.float32, device=device)
    tf_azimuth = repeat(tf_azimuth, "i j -> b i j", b=num_steps).clone()
    tf_azimuth[:, :3, :3] = azimuth

    # Generate the transformation for the elevation.
    deg_elevation = np.deg2rad(elevation)
    elevation = R.from_rotvec(np.array([deg_elevation, 0, 0], dtype=np.float32))
    elevation = torch.tensor(elevation.as_matrix())
    tf_elevation = torch.eye(4, dtype=torch.float32, device=device)
    tf_elevation[:3, :3] = elevation

    return tf_azimuth @ tf_elevation @ tf_translation


def plot_point_cloud(
    vertices: Float[Tensor, "batch dim"],
    alpha: float = 0.5,
    max_points: int = 10_000,
    xlim: tuple[float, float] = (-1.0, 1.0),
    ylim: tuple[float, float] = (-1.0, 1.0),
    zlim: tuple[float, float] = (-1.0, 1.0),
):
    """Plot a point cloud."""
    vertices = vertices.cpu()

    batch, dim = vertices.shape

    if batch > max_points:
        vertices = np.random.default_rng().choice(vertices, max_points, replace=False)
    fig = plt.figure(figsize=(6, 6))
    if dim == 2:
        ax = fig.add_subplot(111)
    elif dim == 3:
        ax = fig.add_subplot(111, projection="3d")
        ax.set_zlabel("z")
        ax.set_zlim(zlim)
        ax.view_init(elev=120.0, azim=270)

    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.scatter(*vertices.T, alpha=alpha, marker=",", lw=0.5, s=1, color="black")
    plt.show()
