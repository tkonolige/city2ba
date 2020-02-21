extern crate cgmath;
extern crate city2ba;
extern crate itertools;
extern crate nalgebra as na;
extern crate ply_rs;
extern crate rand;
extern crate structopt;

use cgmath::{Point3, Vector3};
use city2ba::generate::*;
use city2ba::noise::*;
use city2ba::synthetic::*;
use city2ba::*;
use itertools::Itertools;
use ply_rs::ply::{
    Addable, DefaultElement, ElementDef, Ply, Property, PropertyDef, PropertyType, ScalarType,
};
use ply_rs::writer::Writer;
use std::fs::File;
use std::io::BufWriter;
use std::str::FromStr;
use structopt::StructOpt;

// helper to parse Vector3 with structopt
fn parse_vec3(s: &str) -> Result<Vector3<f64>, std::num::ParseFloatError> {
    let mut it = s.split(",").map(|x| f64::from_str(x));
    let x = it.next().unwrap()?;
    let y = it.next().unwrap()?;
    let z = it.next().unwrap()?;
    Ok(Vector3::new(x, y, z))
}

#[derive(StructOpt, Debug)]
struct PLYOpt {
    /// Input bundle adjustment file in .bal or .bbal format.
    #[structopt(name = "FILE", parse(from_os_str))]
    input: std::path::PathBuf,

    /// Output file in .ply format.
    #[structopt(name = "OUT", parse(from_os_str))]
    out: std::path::PathBuf,
}

#[derive(StructOpt, Debug)]
struct GenerateOpt {
    /// Input .obj model. Y is up and -Z is forward.
    #[structopt(name = "FILE", parse(from_os_str))]
    input: std::path::PathBuf,

    /// Upper bound on the number of cameras to generate.
    #[structopt(long = "cameras", default_value = "100")]
    num_cameras: usize,

    /// Start of range for camera intrinsics. Generated cameras with have intrinsics in the range
    /// [<intrinsics-start>, <intrinsics-end>).
    #[structopt(
            long = "intrinsics-start",
            default_value = "1,0,0",
            parse(try_from_str = parse_vec3)
        )]
    intrinsics_start: Vector3<f64>,

    /// End of range for camera intrinsics.
    #[structopt(
                long = "intrinsics-end",
                default_value = "1,0,0",
                parse(try_from_str = parse_vec3)
            )]
    intrinsics_end: Vector3<f64>,

    /// Upper bound on the number of points visible in the world.
    /// Often, the number of generated points is smaller than this amount.
    #[structopt(long = "points", default_value = "1000")]
    num_world_points: usize,

    /// Maximum distance between a camera and a point.
    #[structopt(long = "max-dist", default_value = "100")]
    max_dist: f64,

    /// Minimum absolute height for cameras generated with Poisson disk sampling. This is an offset
    /// from the bottom of the bounding box of the model.
    #[structopt(long = "ground", default_value = "0", allow_hyphen_values = true)]
    ground: f64,

    /// Height off of terrain for cameras generated with Poisson disk sampling. Cameras are pushed
    /// this far above the surface.
    #[structopt(long = "height", default_value = "1")]
    height: f64,

    /// Do not compute the largest connected component of the camera-point visibility graph. This
    /// may result in problems that have disconnected components.
    #[structopt(long = "no-lcc")]
    no_lcc: bool,

    /// Move .obj model so that its top right corner is at the origin.
    #[structopt(long = "move-to-origin")]
    move_to_origin: bool,

    /// Output file. Will be output in binary format if the ending is .bbal.
    #[structopt(name = "OUT", parse(from_os_str))]
    bal_out: std::path::PathBuf,

    /// Generate cameras randomly along the path named <PATH>. Cameras with face in the direction
    /// of the path. Replaces Poisson disk camera generation
    #[structopt(long = "path", conflicts_with = "ground")]
    path: Option<String>,

    /// If > 0, cameras will be generated sequentially on the path at <step-size> intervals.
    #[structopt(long = "step-size", default_value = "0")]
    step_size: f64,
}

#[derive(StructOpt, Debug)]
struct SyntheticOpt {
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

#[derive(StructOpt, Debug)]
struct NoiseOpt {
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

    /// Standard deviation of translational drift added to the problem. Drift is proportional to
    /// the distance of each camera and point from the origin. Drift is scaled relative to problem
    /// size.
    #[structopt(long = "drift-std", default_value = "0.0")]
    drift_std: f64,

    /// Strength of translational drift added to each camera and point. Constant factor multiplied
    /// with the distance of each camera and point from the origin.
    #[structopt(long = "drift-strength", default_value = "0.0")]
    drift_strength: f64,

    /// Do not scale drift by problem size.
    #[structopt(long = "fixed-drift")]
    fixed_drift: bool,

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

    /// Add noise to the problem that looks like a sin wave over the domain. Cameras are displaced
    /// upwards as sin of normalize distance of the camera/point from the origin.
    #[structopt(long = "sin-strength", default_value = "0.0")]
    sin_strength: f64,

    /// Controls the frequency of the sin wave noise. 1.0 indicates one full sin wave over the
    /// domain (0-pi).
    #[structopt(long = "sin-frequency", default_value = "1.0")]
    sin_frequency: f64,

    /// Output file name. Can output in .bal or .bbal format.
    #[structopt(name = "OUT", parse(from_os_str))]
    output: std::path::PathBuf,
}

#[derive(StructOpt, Debug)]
#[structopt(
    name = "city2ba",
    about = "Tools for generating synthetic bundle adjustment problems."
)]
enum Opt {
    /// Convert a .bal or .bbal to a .ply for visualization.
    PLY(PLYOpt),
    /// Generate a synthetic bundle adjustment problem from a 3D model.
    Generate(GenerateOpt),
    /// Generate a synthetic bundle adjustment problem from an grid of city blocks.
    Synthetic(SyntheticOpt),
    /// Add noise to a bundle adjustment problem.
    Noise(NoiseOpt),
}

fn run_noise(opt: NoiseOpt) -> Result<(), city2ba::Error> {
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

    if opt.fixed_drift {
        let std_dir = bal.std();
        bal = add_drift(
            bal,
            opt.drift_strength,
            opt.drift_angle,
            opt.drift_std,
            std_dir,
        );
    } else {
        bal = add_drift_normalized(bal, opt.drift_strength, opt.drift_angle, opt.drift_std);
    }
    // add sin noise that moves cameras upwards (in positive y)
    if opt.sin_strength > 0. {
        bal = add_sin_noise(
            bal,
            Vector3::new(1., 0., 0.),
            Vector3::new(0., 1., 0.),
            opt.sin_strength,
            opt.sin_frequency,
        );
        bal = add_sin_noise(
            bal,
            Vector3::new(0., 0., 1.),
            Vector3::new(0., 1., 0.),
            opt.sin_strength,
            opt.sin_frequency,
        );
    }
    bal = add_noise(
        bal,
        opt.translation_std,
        opt.rotation_std,
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

/// Write camera locations out to a ply file. Cameras are red, points are green.
fn write_cameras<C: Camera>(
    path: &std::path::Path,
    cameras: &Vec<C>,
    points: &Vec<Point3<f64>>,
    observations: &Vec<Vec<(usize, (f64, f64))>>,
) -> Result<(), std::io::Error> {
    let mut ply = Ply::<DefaultElement>::new();
    let mut point_element = ElementDef::new("vertex".to_string());
    let p = PropertyDef::new("x".to_string(), PropertyType::Scalar(ScalarType::Float));
    point_element.properties.add(p);
    let p = PropertyDef::new("y".to_string(), PropertyType::Scalar(ScalarType::Float));
    point_element.properties.add(p);
    let p = PropertyDef::new("z".to_string(), PropertyType::Scalar(ScalarType::Float));
    point_element.properties.add(p);
    let p = PropertyDef::new("red".to_string(), PropertyType::Scalar(ScalarType::UChar));
    point_element.properties.add(p);
    let p = PropertyDef::new("green".to_string(), PropertyType::Scalar(ScalarType::UChar));
    point_element.properties.add(p);
    let p = PropertyDef::new("blue".to_string(), PropertyType::Scalar(ScalarType::UChar));
    point_element.properties.add(p);
    ply.header.elements.add(point_element);
    let mut edge_element = ElementDef::new("edge".to_string());
    edge_element.properties.add(PropertyDef::new(
        "vertex1".to_string(),
        PropertyType::Scalar(ScalarType::Int),
    ));
    edge_element.properties.add(PropertyDef::new(
        "vertex2".to_string(),
        PropertyType::Scalar(ScalarType::Int),
    ));
    ply.header.elements.add(edge_element);

    // Add first point
    let mut cs: Vec<_> = cameras
        .iter()
        .map(|camera| {
            let mut point = DefaultElement::new();
            point.insert("x".to_string(), Property::Float(camera.center()[0] as f32));
            point.insert("y".to_string(), Property::Float(camera.center()[1] as f32));
            point.insert("z".to_string(), Property::Float(camera.center()[2] as f32));
            point.insert("red".to_string(), Property::UChar(255));
            point.insert("green".to_string(), Property::UChar(0));
            point.insert("blue".to_string(), Property::UChar(0));
            point
        })
        .collect();
    let pts = points.iter().map(|point| {
        let mut p = DefaultElement::new();
        p.insert("x".to_string(), Property::Float(point[0] as f32));
        p.insert("y".to_string(), Property::Float(point[1] as f32));
        p.insert("z".to_string(), Property::Float(point[2] as f32));
        p.insert("red".to_string(), Property::UChar(0));
        p.insert("green".to_string(), Property::UChar(255));
        p.insert("blue".to_string(), Property::UChar(0));
        p
    });
    cs.extend(pts);
    ply.payload.insert("vertex".to_string(), cs);

    let edges = observations
        .iter()
        .enumerate()
        .flat_map(|(ci, obs)| {
            obs.iter().map(move |(pi, _)| {
                let mut e = DefaultElement::new();
                e.insert("vertex1".to_string(), Property::Int(ci as i32));
                e.insert(
                    "vertex2".to_string(),
                    Property::Int((*pi + cameras.len()) as i32),
                );
                e
            })
        })
        .collect();
    ply.payload.insert("edge".to_string(), edges);

    let mut file = BufWriter::new(File::create(path)?);
    let writer = Writer::new();
    writer.write_ply(&mut file, &mut ply).map(|_| ())
}

fn run_ply(opt: PLYOpt) -> std::result::Result<(), city2ba::Error> {
    let bal = BAProblem::from_file(&opt.input)?;
    write_cameras(&opt.out, &bal.cameras, &bal.points, &bal.vis_graph)?;
    Ok(())
}

fn run_synthetic(opt: SyntheticOpt) -> Result<(), city2ba::Error> {
    let ba = synthetic_grid(
        opt.num_cameras_per_block,
        opt.num_points_per_block,
        opt.num_blocks,
        opt.block_length,
        opt.block_inset,
        opt.camera_height,
        opt.point_height,
        opt.max_dist,
        true,
    );
    println!("{}", ba);
    ba.write(&opt.output)?;
    Ok(())
}

fn run_generate(opt: GenerateOpt) -> Result<(), city2ba::Error> {
    let city_obj = tobj::load_obj(&opt.input);
    let (mut models, _) = city_obj.unwrap();

    let model_path = if let Some(path) = opt.path {
        let i = models.iter().position(|x| x.name == path);
        let model_path = match i {
            Some(j) => Some(models[j].clone()),
            None => {
                let names = models.iter().map(|x| x.name.clone()).join(", ");
                panic!(
                    "Could not find a path named {}. Available model names are {}",
                    path, names
                );
            }
        };
        models.retain(|m| m.name != path);
        model_path
    } else {
        None
    };

    if opt.move_to_origin {
        models = move_to_origin(models);
    };

    let dev = embree_rs::Device::new();
    let mut scene = embree_rs::Scene::new(&dev);
    for model in models.iter() {
        let mesh = model_to_geometry(model, &dev);
        scene.attach_geometry(mesh);
    }
    let cscene = scene.commit();

    let mut cameras = if let Some(m_path) = model_path {
        if opt.step_size <= 0.0 {
            generate_cameras_path(&cscene, &m_path, opt.num_cameras)
        } else {
            generate_cameras_path_step(&cscene, &m_path, opt.num_cameras, opt.step_size)
        }
    } else {
        generate_cameras_poisson(&cscene, opt.num_cameras, opt.height, opt.ground)
    };
    println!("Generated {} cameras", cameras.len());

    modify_intrinsics(&mut cameras, opt.intrinsics_start, opt.intrinsics_end);
    println!("Modified intrinsics");

    let points =
        generate_world_points_uniform(&models, &cameras, opt.num_world_points, opt.max_dist);
    println!("Generated {} world points", points.len());

    // TODO: use something more sophisticated to calculate the max distance
    let vis_graph = visibility_graph(&cscene, &cameras, &points, opt.max_dist, true);
    println!(
        "Computed visibility graph with {} edges",
        vis_graph.iter().map(|x| x.len()).sum::<usize>()
    );
    let bal = BAProblem {
        cameras: cameras,
        points: points,
        vis_graph: vis_graph,
    };

    // TODO: sort cameras/points by xy location to have a less random matrix?

    // Remove cameras that view too few points and points that are viewed by too few cameras.
    let bal_lcc = if !opt.no_lcc { bal.cull() } else { bal };
    if bal_lcc.num_cameras() == 0 || bal_lcc.num_points() == 0 {
        return Err(city2ba::Error::EmptyProblem(
            "No cameras remain".to_string(),
        ));
    }
    println!(
        "Computed LCC with {} cameras, {} points, {} edges",
        bal_lcc.num_cameras(),
        bal_lcc.num_points(),
        bal_lcc.num_observations(),
    );

    println!(
        "Total reprojection error: {}",
        bal_lcc.total_reprojection_error(1.)
    );

    bal_lcc.write(&opt.bal_out)?;

    Ok(())
}

fn main() -> Result<(), city2ba::Error> {
    match Opt::from_args() {
        Opt::Generate(opt) => run_generate(opt),
        Opt::Noise(opt) => run_noise(opt),
        Opt::Synthetic(opt) => run_synthetic(opt),
        Opt::PLY(opt) => run_ply(opt),
    }
}
