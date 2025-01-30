from jaxtyping import install_import_hook

# Add runtime type checking to all imports.
with install_import_hook(("src",), ("beartype", "beartype")):
    from src.provided_code import get_bunny, plot_point_cloud

if __name__ == "__main__":
    vertices, _ = get_bunny()
    plot_point_cloud(
        vertices,
        xlim=(-1.0, 1.0),
        ylim=(-1.0, 1.0),
        zlim=(-1.0, 1.0),
    )
