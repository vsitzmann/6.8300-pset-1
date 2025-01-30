from pathlib import Path

from jaxtyping import install_import_hook

# Add runtime type checking to all imports.
with install_import_hook(("src",), ("beartype", "beartype")):
    from src.provided_code import get_bunny, save_image
    from src.puzzle import convert_dataset, load_dataset
    from src.rendering import render_point_cloud


# Put the path to your puzzle here.
DATASET_PATH = Path("puzzle")

if __name__ == "__main__":
    original_dataset = load_dataset(DATASET_PATH)
    converted_dataset = convert_dataset(original_dataset)

    # Render the bunny using the converted camera metadata.
    vertices, faces = get_bunny()
    images = render_point_cloud(
        vertices,
        converted_dataset["extrinsics"],
        converted_dataset["intrinsics"],
    )

    # Save the resulting images.
    for index, image in enumerate(images):
        save_image(image, f"outputs/2_puzzle/view_{index:0>2}.png")
