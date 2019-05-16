extern crate structopt;
use structopt::StructOpt;

extern crate rand;
use rand::distributions::{Distribution, Normal, WeightedIndex};
use rand::Rng;

extern crate city2bal;
use city2bal::*;

extern crate cgmath;
use cgmath::*;

extern crate itertools;
use itertools::Itertools;

#[derive(StructOpt, Debug)]
#[structopt(name = "noise", about = "Tool to add noise/error to a BAL problem")]
struct Opt {
    // Input BAL file
    #[structopt(name = "FILE", parse(from_os_str))]
    input: std::path::PathBuf,

    #[structopt(long = "rotation-std", default_value = "0.0")]
    rotation_std: f32,

    #[structopt(long = "translation-std", default_value = "0.0")]
    translation_std: f32,

    #[structopt(long = "point-std", default_value = "0.0")]
    point_std: f32,

    #[structopt(long = "observation-std", default_value = "0.0")]
    observation_std: f32,

    #[structopt(long = "intrinsic-std", default_value = "0.0")]
    intrinsic_std: f32,

    #[structopt(long = "drift-std", default_value = "0.0")]
    drift_std: f32,

    #[structopt(long = "drift-strength", default_value = "0.0")]
    drift_strength: f32,

    // Probability of a mismatch occurring in a match
    #[structopt(long = "mismatch-chance", default_value = "0.0")]
    mismatch_chance: f32,

    #[structopt(name = "OUT", parse(from_os_str))]
    output: std::path::PathBuf,
}

fn unit_random() -> Vector3<f32> {
    let r = Normal::new(0.0, 1.0);
    Vector3::new(
        r.sample(&mut rand::thread_rng()) as f32,
        r.sample(&mut rand::thread_rng()) as f32,
        r.sample(&mut rand::thread_rng()) as f32,
    )
    .normalize()
}

fn add_drift(bal: BALProblem, strength: f32, std: f32) -> BALProblem {
    // Choose the drift direction to be in line with the largest standard deviation.
    let dir = bal.std().normalize();

    let origin = bal
        .cameras
        .iter()
        .map(|c| &c.loc)
        .chain(bal.points.iter())
        .fold1(|x, y| if x.magnitude() < y.magnitude() { x } else { y })
        .unwrap();

    let r = Normal::new(1.0, std.into());
    let bal_std = bal.std().magnitude();

    let drift_noise = |x: &Vector3<f32>| {
        let distance = (x - origin).magnitude();
        let v = r.sample(&mut rand::thread_rng()) as f32;
        x + dir * strength * v * bal_std * distance
    };
    let cameras = bal
        .cameras
        .iter()
        .map(|c| {
            let mut new_camera = c.clone();
            new_camera.loc = drift_noise(&c.loc);
            new_camera
        })
        .collect();

    let points = bal.points.iter().map(|p| drift_noise(&p)).collect();

    BALProblem {
        cameras: cameras,
        points: points,
        vis_graph: bal.vis_graph,
    }
}

fn add_noise(
    bal: BALProblem,
    translation_std: f32,
    rotation_std: f32,
    intrinsics_std: f32,
    point_std: f32,
    observations_std: f32,
) -> BALProblem {
    let n_translation = Normal::new(0.0, translation_std.into());
    let n_rotation = Normal::new(0.0, rotation_std.into());
    let n_intrinsics = Normal::new(0.0, intrinsics_std.into());
    let n_point = Normal::new(0.0, point_std.into());
    let n_observations = Normal::new(0.0, observations_std.into());
    let bal_std = bal.std().magnitude();

    let cameras = bal
        .cameras
        .iter()
        .map(|c| {
            let dir = c.dir + unit_random() * n_rotation.sample(&mut rand::thread_rng()) as f32;
            let loc = c.loc
                + unit_random() * bal_std * n_translation.sample(&mut rand::thread_rng()) as f32;
            let intrin =
                c.intrin + unit_random() * n_intrinsics.sample(&mut rand::thread_rng()) as f32;
            Camera {
                dir: dir,
                loc: loc,
                intrin: intrin,
                img_size: c.img_size,
            }
        })
        .collect();

    let points = bal
        .points
        .iter()
        .map(|p| p + unit_random() * n_point.sample(&mut rand::thread_rng()) as f32)
        .collect();

    let observations = bal
        .vis_graph
        .iter()
        .map(|obs| {
            obs.iter()
                .map(|(i, (x, y))| {
                    // random direction
                    let n = Normal::new(0.0, 1.0);
                    let nx = n.sample(&mut rand::thread_rng()) as f32;
                    let ny = n.sample(&mut rand::thread_rng()) as f32;
                    let m = (nx.powf(2.0) + ny.powf(2.0)).sqrt();
                    let r = n_observations.sample(&mut rand::thread_rng()) as f32;
                    let x_ = x + nx / m * r;
                    let y_ = y + ny / m * r;
                    (i.clone(), (x_, y_))
                })
                .collect::<Vec<(usize, (f32, f32))>>()
        })
        .collect();

    BALProblem {
        cameras: cameras,
        points: points,
        vis_graph: observations,
    }
}

/// Add incorrect correspondences by swapping two nearby observations.
fn add_incorrect_correspondences(bal: BALProblem, mismatch_chance: f32) -> BALProblem {
    let observations = bal
        .vis_graph
        .into_iter()
        .map(|mut obs| {
            let mut rng = rand::thread_rng();
            for i in 0..obs.len() {
                // Check if we should swap this entry
                if rng.gen_range(0.0, 1.0) <= mismatch_chance {
                    // distance from this observation to all others
                    let mut dists = obs
                        .iter()
                        .map(|(_, (x, y))| {
                            (((obs[i].1).0 - x).powf(2.0) + ((obs[i].1).1 - y).powf(2.0)).sqrt()
                        })
                        .collect::<Vec<_>>();
                    // TODO: use max?
                    dists[i] = 10000000000.0;
                    let weights = dists.iter().map(|x| 1.0 / x).collect::<Vec<_>>();
                    let j = WeightedIndex::new(weights).unwrap().sample(&mut rng);

                    // swap feature indices
                    let tmp = obs[i].0;
                    obs[i].0 = obs[j].0;
                    obs[j].0 = tmp;
                }
            }
            obs
        })
        .collect();

    BALProblem {
        cameras: bal.cameras,
        points: bal.points,
        vis_graph: observations,
    }
}

fn main() -> Result<(), Error> {
    let opt = Opt::from_args();

    let mut bal = BALProblem::from_file(&opt.input)?;

    bal = add_drift(bal, opt.drift_strength, opt.drift_std);
    bal = add_noise(
        bal,
        opt.translation_std,
        opt.rotation_std,
        opt.intrinsic_std,
        opt.point_std,
        opt.observation_std,
    );
    bal = add_incorrect_correspondences(bal, opt.mismatch_chance);

    bal.write(&opt.output).map_err(Error::from)
}
