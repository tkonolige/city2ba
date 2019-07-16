extern crate cgmath;
extern crate embree_rs;
extern crate indicatif;
extern crate itertools;
extern crate nalgebra as na;
extern crate ply_rs;
extern crate poisson;
extern crate rand;
extern crate rayon;
extern crate structopt;
extern crate tobj;

use itertools::Itertools;
use rand::distributions::{Distribution, WeightedIndex};
use rand::{thread_rng, Rng};
use structopt::StructOpt;

use indicatif::{ParallelProgressIterator, ProgressBar, ProgressStyle};

use ply_rs::ply::{
    Addable, DefaultElement, ElementDef, Ply, Property, PropertyDef, PropertyType, ScalarType,
};
use ply_rs::writer::Writer;

use std::convert::TryInto;
use std::fs::File;
use std::io::BufWriter;

use cgmath::prelude::*;
use cgmath::{Basis3, ElementWise, InnerSpace, Point3, Vector3, Vector4};

use rayon::prelude::*;

extern crate city2bal;
use city2bal::*;

#[derive(StructOpt, Debug)]
#[structopt(name = "basic")]
struct Opt {
    #[structopt(name = "FILE", parse(from_os_str))]
    input: std::path::PathBuf,

    #[structopt(long = "cameras", default_value = "100")]
    num_cameras: usize,

    #[structopt(long = "points", default_value = "1000")]
    num_world_points: usize,

    #[structopt(long = "max-dist", default_value = "100")]
    max_dist: f64,

    #[structopt(long = "ground", default_value = "10")]
    ground: f64,

    #[structopt(long = "no-lcc")]
    no_lcc: bool,

    #[structopt(long = "normalize")]
    normalize: bool,

    #[structopt(long = "ply", parse(from_os_str))]
    ply_out: Option<std::path::PathBuf>,

    #[structopt(name = "OUT", parse(from_os_str))]
    bal_out: std::path::PathBuf,

    #[structopt(long = "path")]
    path: Option<String>,
}

fn add_model<'a>(model: &tobj::Model, dev: &'a embree_rs::Device) -> embree_rs::Geometry<'a> {
    let num_tri = model.mesh.indices.len() / 3;
    let num_vert = model.mesh.positions.len() / 3;
    let mut mesh = embree_rs::TriangleMesh::unanimated(dev, num_tri, num_vert);

    {
        let mut verts = mesh.vertex_buffer.map();
        let mut tris = mesh.index_buffer.map();
        for i in 0..num_tri {
            tris[i] = Vector3::new(
                model.mesh.indices[i * 3],
                model.mesh.indices[i * 3 + 1],
                model.mesh.indices[i * 3 + 2],
            );
        }

        for i in 0..num_vert {
            verts[i] = Vector4::new(
                model.mesh.positions[i * 3],
                model.mesh.positions[i * 3 + 1],
                model.mesh.positions[i * 3 + 2],
                0.0,
            );
        }
    }

    let mut geo = embree_rs::Geometry::Triangle(mesh);
    geo.commit();
    geo
}

/// Generate cameras along a path. Cameras will be pointed along the direction of movement of the
/// path.
fn generate_cameras_path(
    _scene: &embree_rs::CommittedScene,
    path: &tobj::Model,
    num_cameras: usize,
) -> Vec<Camera> {
    let vertices = path
        .mesh
        .positions
        .iter()
        .tuples()
        .map(|(x, y, z)| Point3::new(*x as f64, *y as f64, *z as f64))
        .collect::<Vec<_>>();
    let paths = path
        .mesh
        .indices
        .iter()
        .tuples()
        .map(|(i, j)| (vertices[*i as usize], vertices[*j as usize]))
        .collect::<Vec<_>>();
    let path_lengths = paths.iter().map(|(x, y)| (y - x).magnitude());

    let mut rng = rand::thread_rng();
    let dist = rand::distributions::WeightedIndex::new(path_lengths).unwrap();

    dist.sample_iter(&mut rng)
        .take(num_cameras)
        .map(|i| {
            let (x, y) = paths[i];
            let d = thread_rng().gen_range(0.0, 1.0);
            let dir = y - x;

            let pos = x + d * dir;

            Camera::from_position_direction(
                pos,
                Basis3::between_vectors(dir.normalize(), Vector3::new(0.0, 0.0, -1.0)),
                Vector3::new(1.0, 0.0, 0.0),
            )
        })
        .collect()
}

// Generate camera locations by placing cameras on a regular grid throughout the image. Locations
// are then filtered based on their height.
// TODO: smarter pattern for points. choose initial point and expand search?
// TODO: keep generating points until we hit the target number
fn generate_cameras_grid(
    scene: &embree_rs::CommittedScene,
    num_points: usize,
    ground: f64,
) -> Vec<Camera> {
    let mut intersection_ctx = embree_rs::IntersectContext::coherent(); // not sure if this matters
    let mut positions = Vec::new();

    let poisson = poisson::Builder::<f64, na::Vector2<f64>>::with_samples(
        num_points * 2, // x2 seems to get us closer to the desired amount
        1.0,
        poisson::Type::Normal,
    )
    .build(rand::thread_rng(), poisson::algorithm::Ebeida);
    let samples = poisson.generate();

    let bounds = scene.bounds();
    // add a little wiggle room
    let start = Point3::new(
        bounds.upper_x as f64,
        bounds.upper_y as f64,
        bounds.upper_z as f64 + 0.1,
    );
    let delta = Vector3::new(
        (bounds.upper_x - bounds.lower_x) as f64,
        (bounds.upper_y - bounds.lower_y) as f64,
        0.0,
    );

    for sample in samples {
        let origin = start - delta.mul_element_wise(Vector3::new(sample[0], sample[1], 0.0));
        let direction = Vector3::new(0.0, -1.0, 0.0); // looking directly down
        let ray = embree_rs::Ray::new(
            origin.cast::<f32>().unwrap().to_vec(),
            direction.cast::<f32>().unwrap(),
        );
        let mut ray_hit = embree_rs::RayHit::new(ray);
        scene.intersect(&mut intersection_ctx, &mut ray_hit);
        if ray_hit.hit.hit() {
            // push point up a little from where it hit
            let pt = origin + direction * (ray_hit.ray.tfar as f64) + Vector3::new(0.0, 1.0, 0.0);

            if pt[2] < bounds.lower_z as f64 + ground {
                positions.push(pt);
            }
        }
    }

    positions
        .into_iter()
        .map(|position| {
            // choose a random direction looking at the horizon
            // TODO: check that we are not too close to an object
            let dir = thread_rng().gen_range(0.0, 2.0 * std::f64::consts::PI);
            let around_z = Basis3::from_angle_y(cgmath::Rad(dir));
            Camera::from_position_direction(position, around_z, Vector3::new(1.0, 0.0, 0.0))
        })
        .collect::<Vec<_>>()
}

fn iter_triangles<'a>(
    mesh: &'a tobj::Mesh,
) -> impl Iterator<Item = (Point3<f64>, Point3<f64>, Point3<f64>)> + 'a {
    mesh.indices
        .iter()
        .map(move |index| {
            let i: usize = *index as usize;
            Point3::new(
                mesh.positions[i * 3] as f64,
                mesh.positions[i * 3 + 1] as f64,
                mesh.positions[i * 3 + 2] as f64,
            )
        })
        .tuples::<(_, _, _)>()
}

fn get_triangle(mesh: &tobj::Mesh, i: usize) -> (Point3<f64>, Point3<f64>, Point3<f64>) {
    let v = (0..3).map(|j| {
        let i0 = mesh.indices[i * 3 + j] as usize;
        Point3::new(mesh.positions[i0], mesh.positions[i0+1], mesh.positions[i0+2]).cast::<f64>().unwrap()
    }).collect::<Vec<_>>();
    (v[0], v[1], v[2])
}

/// Choose a uniformly distributed random point in the given triangle.
fn random_point_in_triangle(v0: Point3<f64>, v1: Point3<f64>, v2: Point3<f64>) -> Point3<f64> {
    let mut rx = thread_rng().gen_range(0.0, 1.0);
    let mut ry = thread_rng().gen_range(0.0, 1.0);

    // sample from parallelogram, but reflect points outside of the original triangle
    if rx + ry > 1.0 {
        rx = 1.0 - rx;
        ry = 1.0 - ry;
    }

    v0 + rx * (v1 - v0) + ry * (v2 - v0)
}

// TODO: only generate points that are close enough to cameras
fn generate_world_points_poisson(models: &Vec<tobj::Model>, num_points: usize) -> Vec<Point3<f64>> {
    // TODO: filter meshes by distance from cameras
    let areas = models.iter().flat_map(|model| {
        iter_triangles(&model.mesh)
            .map(|(v0, v1, v2)| ((v1 - v0).cross(v2 - v0).magnitude() / 2.0) as f64)
    });
    let indices = models
        .iter()
        .enumerate()
        .flat_map(|(i, model)| iter_triangles(&model.mesh).enumerate().map(move |(j, _)| (i, j)))
        .collect::<Vec<_>>();

    let dist = WeightedIndex::new(areas).unwrap();
    let mut rng = thread_rng();

    let mut points = Vec::with_capacity(num_points);
    while points.len() < num_points {
        let i = dist.sample(&mut rng);
        let (m, j) = indices[i];
        let mesh = &models[m].mesh;
        let (v0, v1, v2) = get_triangle(mesh, j);
        points.push(random_point_in_triangle(v0, v1, v2));
    }

    points
}

fn visibility_graph(
    scene: &embree_rs::CommittedScene,
    cameras: &Vec<Camera>,
    points: &Vec<Point3<f64>>,
    max_dist: f64,
) -> Vec<Vec<(usize, (f64, f64))>> {
    let pb = ProgressBar::new(cameras.len().try_into().unwrap());
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{bar:40}] {percent}% ({eta})")
            .progress_chars("#-"),
    );

    cameras
        .par_iter()
        .progress_with(pb)
        .map(|camera| {
            let mut intersection_ctx = embree_rs::IntersectContext::coherent(); // not sure if this matters

            let mut local_obs = Vec::with_capacity(points.len());
            let mut local_rays = Vec::with_capacity(points.len());
            for (i, point) in points.iter().enumerate() {
                // project point into camera frame
                let p_camera = camera.project_world(point);
                // check if point is infront of camera (camera looks down negative z)
                if (camera.center() - point).magnitude() < max_dist && p_camera.z <= 0.0 {
                    let p = camera.project(p_camera);

                    // check point is in camera frame
                    if p.x >= -1.0 && p.x <= 1.0 && p.y >= -1.0 && p.y < 1.0 {
                        // ray pointing from camera towards the point
                        let dir = point - camera.center();
                        let mut ray = embree_rs::Ray::new(
                            camera.center().cast::<f32>().unwrap().to_vec(),
                            dir.normalize().cast::<f32>().unwrap(),
                        );

                        // Check if there is anything between the camera and the point. We stop a
                        // little short of the point to make sure we don't hit it.
                        ray.tfar = dir.magnitude() as f32 - 1e-4;
                        local_rays.push(ray);
                        local_obs.push((i, (p.x, p.y)));
                    }
                }
            }

            // filter by occluded rays
            scene.occluded_stream_aos(&mut intersection_ctx, &mut local_rays);
            local_rays
                .iter()
                .zip(local_obs)
                .filter(|x| !x.0.tfar.is_infinite())
                .map(|x| x.1)
                .collect()
        })
        .collect()
}

/// Write camera locations out to a ply file. Does not provide color or orientation.
fn write_cameras(
    path: &std::path::Path,
    cameras: &Vec<Camera>,
    points: &Vec<Point3<f64>>,
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

    let cs_proj = cameras
        .iter()
        .map(|camera| {
            let mut point = DefaultElement::new();
            let p = camera.to_world(Point3::new(0.0, 0.0, -1.0));
            point.insert("x".to_string(), Property::Float(p[0] as f32));
            point.insert("y".to_string(), Property::Float(p[1] as f32));
            point.insert("z".to_string(), Property::Float(p[2] as f32));
            point.insert("red".to_string(), Property::UChar(0));
            point.insert("green".to_string(), Property::UChar(0));
            point.insert("blue".to_string(), Property::UChar(255));
            point
        })
        .collect::<Vec<_>>();
    cs.extend(cs_proj);

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

    let mut file = BufWriter::new(File::create(path)?);
    let writer = Writer::new();
    writer.write_ply(&mut file, &mut ply).map(|_| ())
}

fn normalize(models: &Vec<tobj::Model>) -> Vec<tobj::Model> {
    let min_vec = |x: Vector3<f32>, y: Vector3<f32>| {
        Vector3::new(x[0].min(y[0]), x[1].min(y[1]), x[2].min(y[2]))
    };
    let min_locs = models
        .iter()
        .map(|model| {
            model
                .mesh
                .positions
                .chunks(3)
                .map(|chunk| Vector3::new(chunk[0], chunk[1], chunk[2]))
                .fold1(|x, y| min_vec(x, y))
                .unwrap()
        })
        .fold1(|x, y| min_vec(x, y))
        .unwrap();

    models
        .iter()
        .map(|model| {
            let new_positions = model
                .mesh
                .positions
                .chunks(3)
                .map(|chunk| {
                    let x = Vector3::new(chunk[0], chunk[1], chunk[2]) - min_locs;
                    // ignore z for now
                    vec![x[0], x[1], chunk[2]].into_iter()
                })
                .flatten()
                .collect();
            tobj::Model::new(
                tobj::Mesh::new(
                    new_positions,
                    model.mesh.normals.clone(),
                    model.mesh.texcoords.clone(),
                    model.mesh.indices.clone(),
                    model.mesh.material_id,
                ),
                model.name.clone(),
            )
        })
        .collect()
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

    if opt.normalize {
        models = normalize(&models);
    };

    // create embree device
    let dev = embree_rs::Device::new();
    let mut scene = embree_rs::Scene::new(&dev);
    for model in models.iter() {
        let mesh = add_model(model, &dev);
        scene.attach_geometry(mesh);
    }
    let cscene = scene.commit();

    let cameras = if let Some(m_path) = model_path {
        generate_cameras_path(&cscene, &m_path, opt.num_cameras)
    } else {
        generate_cameras_grid(&cscene, opt.num_cameras, opt.ground)
    };
    println!("Generated {} cameras", cameras.len());

    let points = generate_world_points_poisson(&models, opt.num_world_points);
    println!("Generated {} world points", points.len());

    // TODO: use something more sophisticated to calculate the max distance
    let vis_graph = visibility_graph(&cscene, &cameras, &points, opt.max_dist);
    println!(
        "Computed visibility graph with {} edges",
        vis_graph.iter().map(|x| x.len()).sum::<usize>()
    );
    let bal = BALProblem {
        cameras: cameras,
        points: points,
        vis_graph: vis_graph,
    };

    // TODO: sort cameras, points by xy location to have a less random matrix?

    // Remove cameras that view too few points and points that are viewed by too few cameras.
    // let bal_lcc = if !opt.no_lcc { bal.cull() } else { bal };
    let bal_lcc = if !opt.no_lcc {
        bal.remove_singletons()
    } else {
        bal
    };
    println!(
        "Computed LCC with {} cameras, {} points, {} edges",
        bal_lcc.cameras.len(),
        bal_lcc.points.len(),
        bal_lcc.vis_graph.iter().map(|x| x.len()).sum::<usize>()
    );

    println!(
        "Total reprojection error: {}",
        bal_lcc.total_reprojection_error()
    );

    bal_lcc.write(&opt.bal_out)?;

    match opt.ply_out {
        Some(path) => write_cameras(&path, &bal_lcc.cameras, &bal_lcc.points),
        None => Ok(()),
    }
}
