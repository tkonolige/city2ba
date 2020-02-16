extern crate city2bal;
extern crate structopt;
use city2bal::*;
use structopt::StructOpt;

#[derive(StructOpt, Debug)]
#[structopt(
    name = "check",
    about = "Check accuracy of a bundle adjustment problem"
)]
struct Opt {
    #[structopt(name = "FILE", parse(from_os_str))]
    input: std::path::PathBuf,
}

fn main() -> Result<(), Error> {
    let opt = Opt::from_args();

    let ba_problem = BAProblem::from_file(&opt.input)?;
    println!("{}", ba_problem);
    println!(
        "{} total reprojection error",
        ba_problem.total_reprojection_error(1.)
    );
    println!(
        "{} total reprojection error (L2)",
        ba_problem.total_reprojection_error(2.)
    );

    Ok(())
}
