# 6.S980 Homework 1

## Getting Started

**Using Python 3.9 or newer,** create a virtual environment as follows:

```
python3 -m venv venv
source venv/bin/activate
```

Next, install PyTorch using the instructions [here](https://pytorch.org/get-started/locally/). Select pip as the installation method. **If you're on Linux and have a CUDA-capable GPU, select the latest CUDA version.** This will give you a command like this:

```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Finally, install this homework's other dependencies:

```
pip install -r requirements.txt
```

You can now open the project directory in VS Code. Within VS Code, open the command palette (<key>⌘ command</key> <key>⇧ shift</key> <key>P</key>), run `Python: Select Interpreter`, and choose the virtual environment you created in the previous steps.

For the best editing experience, install the following VS Code extensions:

* Python (`ms-python.python`)
* Pylance (`ms-python.vscode-pylance`)
* Black Formatter (`ms-python.black-formatter`)
* Ruff (`charliermarsh.ruff`)

## Project Components

Before getting started: if you're new to PyTorch or `einsum`, you can optionally check out [this Colab notebook](https://drive.google.com/file/d/19RAkdGtmwnM9DcAmx1dF2NgHskZ8i2vl/view?usp=sharing).

### Part 0: Introduction

This part doesn't involve programming. Just run the script using the provided VS Code launch configuration:

- Navigate to the debugger (<key>⌘ command</key> <key>⇧ shift</key> <key>D</key>).
- Select `Part 0: Introduction` from the drop-down menu.
- Click the green run button.

<details>
<summary>Running code directly from a command line (not recommended)</summary>
<br>

Remember to activate your virtual environment using `source venv/bin/activate` first. Then, run the following command:

```
python3 -m scripts.0_introduction
```

</details>

### Part 1: Point Cloud Projection

In this part, you'll have to fill out the following files:

- `src/geometry.py`
- `src/rendering.py`

Check your work by running `Part 1: Projection`. This will write the rendered images to the directory `outputs/1_projection`.

#### Notes on Camera Formats

##### Extrinsics

In this class, camera extrinsics are represented as 4x4 matrices in OpenCV-style camera-to-world format. This means that matrix-vector multiplication between an extrinsics matrix and a point in camera space yields a point in world space.

<img src="data/opencv_coordinate_system.png" alt="OpenCV Coordinate System" width="25%" />

##### Intrinsics

In this class, camera intrinsics are represented as 3x3 matrices that have been normalized via division by the image height $h$ and image width $w$.

$$
K = \begin{bmatrix}
    \frac{f_x}{w} & 0 & \frac{c_x}{w} \\
    0 & \frac{f_y}{h} & \frac{c_y}{h} \\
    0 & 0 & 1
\end{bmatrix}
$$

#### Notes on Type Checking

All of the functions in this homework are annotated with types. These are enforced at runtime using [jaxtyping](https://github.com/google/jaxtyping) and [beartype](https://github.com/beartype/beartype). If you're not familiar with `jaxtyping`, you should read the (short) [jaxtyping documentation](https://docs.kidger.site/jaxtyping/api/array/) to learn the tensor annotation format.

**Hint:** These annotations have important implications for broadcasting. If your code does not support the broadcasting implied by the annotations, you will not get full credit. See the note below for more details.

<details>
<summary>A note on the batch dimension annotations used in the homework.</summary>
<br>

The annotations `*batch` and `*#batch` are used for functions that can handle inputs with arbitrary batch dimensions. They differ in that `*#batch` states that batch dimensions can be broadcasted. For example:

```python
def broadcastable(a: Float[Tensor, "*#batch 4 4"], b: Float[Tensor, "*#batch 4"]) -> Float[Tensor, "*batch"]:
    ...

# This works, because the shapes (1, 2, 3, 1) and (2, 3, 5) can be broadcasted.
broadcastable(
    torch.randn((1, 2, 3, 1, 4, 4)),  # a
    torch.randn((2, 3, 5, 4)), # b
)

def not_broadcastable(a: Float[Tensor, "*batch 4 4"], b: Float[Tensor, "*batch 4"]):
    pass

# This doesn't work, since the shapes (1, 2, 3, 1) and (2, 3, 5) are not exactly the same.
not_broadcastable(
    torch.randn((1, 2, 3, 1, 4, 4)),  # a
    torch.randn((2, 3, 5, 4)), # b
)
```

All functions in `geometry.py` that have multiple parameters use `*#batch`, meaning that you must fill them out to handle broadcasting correctly. The functions `homogenize_points` and `homogenize_vectors` instead use `*batch`, since broadcasting doesn't apply when there's only one parameter. All functions return `*batch`, which means that outputs should have a fully broadcasted shape. In the example above, the output of `broadcastable` would have shape `(1, 2, 3, 5)`.

</details>



#### Running Tests

We've included some tests that you can use as a sanity check. These tests only test basic functionality—it's up to you to ensure that your code can handle more complex use cases (e.g. batch dimensions, corner cases). Run these tests from the project root directory as follows:

```
python -m pytest tests
```

You can also run the tests from inside VS Code's graphical interface. From the command palette (<key>⌘ command</key> <key>⇧ shift</key> <key>P</key>), run `Testing: Focus on Test Explorer View`.

### Part 2: Dataset Puzzle

In this part of the homework, you're given a synthetic computer vision dataset in which the camera format has not been documented. You must convert between this unknown format and the OpenCV format described in part 1. To do so, fill in the functions in `src/puzzle.py`.

Each student will receive a unique dataset with a randomly generated format. Per-student dataset download links will be made available via a Piazza announcement.

#### Dataset Format

Each dataset contains 32 images and a `metadata.json` file. The `metadata.json` file contains two keys:

* `intrinsics`: These intrinsics use the normalized format described in part 1.
* `extrinsics`: These extrinsics use a randomized format described below.

The extrinsics are either in camera-to-world format or world-to-camera format. The axes have been randomized, meaning that the camera look, up, and right vectors could be any of $(+x, -x, +y, -y, +z, -z)$. Note that this could yield a left-handed coordinate system! Here's an example of Blender's (right-handed) camera coordinate system:

<img src="data/blender_coordinate_system.png" alt="Blender Coordinate System" width="25%" />

#### Dataset Camera Arrangement

The cameras are arranged as described below. Use this information to help you figure out your camera format.

* The camera origins are always exactly 2 units from the origin.
* The world up vector is $+y$, and all cameras have $y \geq 0$.
* All camera look vectors point directly at the origin.
* All camera up vectors are pointed "up" in the world. In other words, the dot product between any camera up vector and $+y$ is positive.

Hint: How might one build a rotation matrix to convert between camera coordinate systems?

#### Checking Your Work

Run the script for part 2 to check your work. If your conversion function works correctly, you should be able to exactly reproduce the images in your dataset. If you want to test your `load_dataset` function in isolation, you can use the dataset located at `data/sample_dataset` with a `convert_dataset` that simply returns its input.

Note that you may use the mesh at `data/stanford_bunny.obj` if you find it helpful to do so, although using it is not required to find the solution.

## Collaboration Policy

You may work with other students and use AI tools (e.g. ChatGPT, GitHub Copilot), but must submit code that you understand fully and have written yourself.

## Submission Policy

Before submitting, ensure that your code has been formatted using Black and linted using Ruff. Otherwise, you will lose points. Also, double-check that you have not changed any of the function signatures in `geometry.py`, `puzzle.py`, or `rendering.py`. Submit your work on Canvas.

## [Optional] Bonus Problem

Each homework will have a bonus problem that we will use to allocate A+ grades. **These problems are completely optional.** For this homework, the bonus problem is as follows:

Can you devise a way to automatically solve *everyone's* puzzles? Create a script that converts a folder of dataset `.zip` files to a folder of dataset `.zip` files in standardized (converted) format. If you attempt this problem, make sure to include your script's location and your general approach in your answer to `explanation_of_problem_solving_process` in `puzzle.py`.

Also, if you manage to do this, please don't spoil everyone else's puzzles! :)
