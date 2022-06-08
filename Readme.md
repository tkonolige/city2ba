# City2BA
[![docs.rs](https://docs.rs/city2ba/badge.svg)](https://docs.rs/city2ba)

A collection of tools for generating synthetic bundle adjustment datasets.

Datasets can either be generated programatically via the library or using the included executables. When using `SnavelyCamera`s, the coordinate system is -z forward, y up.

```bash
# Generate a problem from a 3D model
city2ba generate test_scene.obj problem.bal --cameras 100 --points 200

# Add noise to the problem
city2ba noise problem.bal problem_noised.bal --drift-strength 0.001 --rotation-std 0.0001

# Generate a problem using a city block grid
city2ba synthetic problem.bal --blocks 4

# Convert a problem to a format for visualization
city2ba ply problem.bal problem.ply
```

## Installation

First install embree (available at [https://github.com/embree/embree](https://github.com/embree/embree)). Then install cargo to build the code and dependencies ([https://rustup.rs](https://rustup.rs) is the easiest way to get cargo).

To install the latest stable version run:
```bash
EMBREE_DIR=/path/to/embree/lib cargo install city2ba
```
Make sure to point `EMBREE_DIR` to the `lib` directory inside of the directory
where you installed embree.

To build the latest version run:
```bash
git clone https://github.com/tkonolige/city2ba.git
cd city2ba
EMBREE_DIR=/path/to/embree/lib cargo install --path .
```
Make sure to point `EMBREE_DIR` to the `lib` directory inside of the directory
where you installed embree.

The `city2ba` binary is installed into `$HOME/.cargo/bin`. In order to run the
binary you will need to add this directory to your `PATH` (you can do this with
`export PATH="$HOME/.cargo/bin:$PATH"`, you may want to add this to your
`.bashrc` or equivalent).

A simple scene is provided in `test_scene.obj` for you to experiment with. You can download it from [https://github.com/tkonolige/city2ba/raw/master/test_scene.obj](https://github.com/tkonolige/city2ba/raw/master/test_scene.obj).

## Development

Build City2BA locally with:
```bash
git clone https://github.com/tkonolige/city2ba.git
cd city2ba
cargo build --release # release mode is recommended for performance
```
Run tests with:
```bash
cargo test
```
Executables can be run with:
```bash
cargo --release ARGS GO HERE
```

## Contributing and Support

Please use the GitHub issue tracker to report issues, ask questions, and submit patches.
