extern crate city2bal;
extern crate structopt;

use structopt::StructOpt;

use city2bal::*;

#[derive(StructOpt, Debug)]
#[structopt(name = "basic")]
struct Opt {
    #[structopt(long = "cameras-per-block", default_value = "10")]
    num_cameras_per_block: usize,

    #[structopt(long = "points-per-block", default_value = "10")]
    num_points_per_block: usize,

    #[structopt(long = "max-dist", default_value = "10")]
    max_dist: f64,

    #[structopt(long = "camera-height", default_value = "1")]
    camera_height: f64,

    #[structopt(long = "point-height", default_value = "1")]
    point_height: f64,

    #[structopt(long = "block-inset", default_value = "1")]
    block_inset: f64,

    #[structopt(long = "block-length", default_value = "20")]
    block_length: f64,

    #[structopt(long = "blocks", default_value = "5")]
    num_blocks: usize,

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
