//! **city2ba** is a set of tools for generating synthetic bundle adjustment datasets.
//!
//! Bundle Adjustment is a nonlinear global optimization used to reduce noise in structure from
//! motion (SfM) and simultaneous localization and mapping applications (SLAM). See
//! [https://en.wikipedia.org/wiki/Bundle_adjustment](https://en.wikipedia.org/wiki/Bundle_adjustment)
//! for more details. Not many bundle adjustment datasets are freely available, so this package
//! contains tools for generating synthetic ones. A bundle adjustment dataset contains a set of
//! cameras, a set of 3D points, and a set of camera-point observations. The goal of a bundle
//! adjuster is to minimizer the difference between the projection of each 3D point into each
//! camera and the location where it was actually observed. This package provides ways for
//! generating zero error (ground truth) datasets in the [generate] and [synthetic] modules, and
//! ways to add noise to existing datasets (so they are no longer zero error) in the [noise]
//! module.
//!
//! This crate also provides command line tools for generating bundle adjustment datasets:
//! ```bash
//! # Generate a problem from a 3D model
//! city2ba generate model.obj problem.bal --num-cameras 100 --num-points 200
//!
//! # Add noise to the problem
//! city2ba noise problem.bal problem_noised.bal --drift-strength 0.001 --rotation-std 0.0001
//!
//! # Generate a problem using a city block grid
//! city2ba synthetic problem.bal --blocks 4
//!
//! # Convert a problem to a format for visualization
//! city2ba ply problem.bal problem.ply
//! ```

#[macro_use]
extern crate itertools;

mod baproblem;
pub mod generate;
pub mod noise;
pub mod synthetic;

pub use baproblem::*;
