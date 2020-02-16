extern crate city2ba;
extern crate structopt;

use structopt::StructOpt;

use city2ba::*;

#[derive(StructOpt, Debug)]
#[structopt(
    name = "synthetic",
    about = "Generate a synthetic test problem with a city-block structure."
)]
struct Opt {
    /// Number of cameras per block. Number of total cameras generated is <cameras-per-block> *
    /// <blocks> * 2.
    #[structopt(long = "cameras-per-block", default_value = "10")]
    num_cameras_per_block: usize,

    /// Number of points per block. Number of total points generated is <points-per-block> *
    /// <blocks> * 2.
    #[structopt(long = "points-per-block", default_value = "10")]
    num_points_per_block: usize,

    /// Maximum viewing distance of a point by a camera.
    #[structopt(long = "max-dist", default_value = "10")]
    max_dist: f64,

    /// Height of cameras placed in the world.
    #[structopt(long = "camera-height", default_value = "1")]
    camera_height: f64,

    /// Height of points placed in the world.
    #[structopt(long = "point-height", default_value = "1")]
    point_height: f64,

    /// Inset of points from the edges of each grid cell.
    #[structopt(long = "block-inset", default_value = "1")]
    block_inset: f64,

    /// Length of each grid cell.
    #[structopt(long = "block-length", default_value = "20")]
    block_length: f64,

    /// Number of blocks in the grid.
    #[structopt(long = "blocks", default_value = "5")]
    num_blocks: usize,

    /// Output file in .bal or .bbal format.
    #[structopt(name = "OUTPUT", parse(from_os_str))]
    output: std::path::PathBuf,
}

fn main() -> Result<(), std::io::Error> {
    let opt = Opt::from_args();
    let ba = synthetic_grid(
        opt.num_cameras_per_block,
        opt.num_points_per_block,
        opt.block_inset,
        opt.num_blocks,
        opt.block_length,
        opt.camera_height,
        opt.point_height,
        opt.max_dist,
    );
    println!("{}", ba);
    ba.write(&opt.output)
}
