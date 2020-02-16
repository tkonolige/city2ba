extern crate city2bal;
extern crate itertools;
extern crate nalgebra as na;
extern crate rand;
extern crate structopt;

use itertools::Itertools;
use structopt::StructOpt;

use std::str::FromStr;

use city2bal::*;

// helper to parse Vector3 with structopt
fn parse_vec3(s: &str) -> Result<Vector3<f64>, std::num::ParseFloatError> {
    let mut it = s.split(",").map(|x| f64::from_str(x));
    let x = it.next().unwrap()?;
    let y = it.next().unwrap()?;
    let z = it.next().unwrap()?;
    Ok(Vector3::new(x, y, z))
}

#[derive(StructOpt, Debug)]
#[structopt(
    name = "city2bal",
    about = "Generate a bundle adjustment problem from an .obj file."
)]
struct Opt {
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

fn main() -> Result<(), std::io::Error> {
    let opt = Opt::from_args();

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

    // create embree device
    let dev = embree_rs::Device::new();
    let mut scene = embree_rs::Scene::new(&dev);
    for model in models.iter() {
        let mesh = model2geometry(model, &dev);
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
    let vis_graph = visibility_graph(&cscene, &cameras, &points, opt.max_dist);
    println!(
        "Computed visibility graph with {} edges",
        vis_graph.iter().map(|x| x.len()).sum::<usize>()
    );
    let bal = BAProblem {
        cameras: cameras,
        points: points,
        vis_graph: vis_graph,
    };

    // TODO: sort cameras, points by xy location to have a less random matrix?

    // Remove cameras that view too few points and points that are viewed by too few cameras.
    let bal_lcc = if !opt.no_lcc { bal.cull() } else { bal };
    println!(
        "Computed LCC with {} cameras, {} points, {} edges",
        bal_lcc.cameras.len(),
        bal_lcc.points.len(),
        bal_lcc.vis_graph.iter().map(|x| x.len()).sum::<usize>()
    );

    println!(
        "Total reprojection error: {}",
        bal_lcc.total_reprojection_error(1.)
    );

    bal_lcc.write(&opt.bal_out)?;

    Ok(())
}
