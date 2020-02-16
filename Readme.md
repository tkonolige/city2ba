# City2BA

A collection of tools for generating synthetic bundle adjustment datasets.

Datasets can either be generated programatically via the library or using the included executables.

```bash
# Generate a problem from a 3D model
city2ba generate model.obj problem.bal --num-cameras 100 --num-points 200

# Add noise to the problem
city2ba noise problem.bal problem_noised.bal --drift-strength 0.001 --rotation-std 0.0001

# Generate a problem using a city block grid
city2ba synthetic problem.bal --blocks 4

# Convert a problem to a format for visualization
city2ba ply problem.bal problem.ply
```
