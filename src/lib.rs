//! Tools for generating synthetic bundle adjustment datasets.

#[macro_use]
extern crate itertools;

mod baproblem;
pub mod generate;
pub mod noise;
pub mod synthetic;

pub use baproblem::*;
