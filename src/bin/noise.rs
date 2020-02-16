extern crate structopt;
use structopt::StructOpt;

extern crate city2ba;
use city2ba::*;

#[derive(StructOpt, Debug)]
#[structopt(name = "noise", about = "Tool to add noise/error to a BA problem")]
struct Opt {
    /// Input bundle adjustment problem. Should be in .bal or .bbal file format.
    #[structopt(name = "FILE", parse(from_os_str))]
    input: std::path::PathBuf,

    /// Standard deviation of Gaussian noise added to camera rotations.
    #[structopt(long = "rotation-std", default_value = "0.0")]
    rotation_std: f64,

    /// Standard deviation of Gaussian noise added to camera translations.
    #[structopt(long = "translation-std", default_value = "0.0")]
    translation_std: f64,

    /// Standard deviation of Gaussian noise added to point translations.
    #[structopt(long = "point-std", default_value = "0.0")]
    point_std: f64,

    /// Standard deviation of Gaussian noise added to observations.
    #[structopt(long = "observation-std", default_value = "0.0")]
    observation_std: f64,

    /// Standard deviation of Gaussian noise added to camera intrinsics. Treats all intrinsics as
    /// if they are at the same scale.
    #[structopt(long = "intrinsic-std", default_value = "0.0")]
    intrinsic_std: f64,

    /// Standard deviation of translational drift added to the problem. Drift is proportional to
    /// the distance of each camera and point from the origin. Drift is scaled relative to problem
    /// size.
    #[structopt(long = "drift-std", default_value = "0.0")]
    drift_std: f64,

    /// Strength of translational drift added to each camera and point. Constant factor multiplied
    /// with the distance of each camera and point from the origin.
    #[structopt(long = "drift-strength", default_value = "0.0")]
    drift_strength: f64,

    /// Strength of rotational drift added to each camera and point.
    #[structopt(long = "drift-angle", default_value = "0.0")]
    drift_angle: f64,

    /// Probability of turning a correct correspondence into a incorrect one.
    #[structopt(long = "mismatch-chance", default_value = "0.0")]
    mismatch_chance: f64,

    /// Percentage of features to keep per camera.
    #[structopt(long = "drop-features", default_value = "1.0")]
    drop_features: f64,

    /// Percentage of landmarks to split in two separate landmarks at the same location.
    /// Observations will be split between the two.
    #[structopt(long = "split-landmarks", default_value = "0.0")]
    split_landmarks: f64,

    /// Percentage of observations that sees two landmarks as the same one.
    #[structopt(long = "join-landmarks", default_value = "0.0")]
    join_landmarks: f64,

    /// Output file name. Can output in .bal or .bbal format.
    #[structopt(name = "OUT", parse(from_os_str))]
    output: std::path::PathBuf,
}

fn main() -> Result<(), Error> {
    let opt = Opt::from_args();

    let mut bal = BAProblem::from_file(&opt.input)?;

    println!(
        "Initial error: {:.2e} (L1) {:.2e} (L2)",
        bal.total_reprojection_error(1.),
        bal.total_reprojection_error(2.)
    );

    if opt.drop_features < 1.0 {
        bal = drop_features(bal, opt.drop_features);
        bal = bal.cull();
    }

    // Join before splitting so that we don't accidentally join two split landmarks
    if opt.join_landmarks > 0.0 {
        bal = join_landmarks(bal, opt.split_landmarks);
        bal = bal.cull();
    }

    if opt.split_landmarks > 0.0 {
        bal = split_landmarks(bal, opt.split_landmarks);
        bal = bal.cull();
    }

    bal = add_drift(bal, opt.drift_strength, opt.drift_angle, opt.drift_std);
    bal = add_noise(
        bal,
        opt.translation_std,
        opt.rotation_std,
        opt.intrinsic_std,
        opt.point_std,
        opt.observation_std,
    );
    bal = add_incorrect_correspondences(bal, opt.mismatch_chance);

    println!(
        "BA Problem with {} cameras, {} points, {} correspondences",
        bal.num_cameras(),
        bal.num_points(),
        bal.num_observations()
    );

    println!(
        "Final error: {:.2e} (L1) {:.2e} (L2)",
        bal.total_reprojection_error(1.),
        bal.total_reprojection_error(2.)
    );

    bal.write(&opt.output).map_err(Error::from)
}
