extern crate city2bal;
extern crate structopt;
use city2bal::*;
use structopt::StructOpt;

#[derive(StructOpt, Debug)]
#[structopt(name = "basic")]
struct Opt {
    #[structopt(name = "FILE", parse(from_os_str))]
    input: std::path::PathBuf,
}

fn main() {
    let opt = Opt::from_args();

    let bal_problem = BALProblem::from_file(&opt.input);
    println!("{} total reprojection error", bal_problem.total_reprojection_error());
}
