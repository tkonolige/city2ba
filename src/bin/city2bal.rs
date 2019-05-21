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
use cgmath::{ElementWise, InnerSpace, Vector3, Vector4};

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
    max_dist: f32,

    #[structopt(long = "ply", parse(from_os_str))]
    ply_out: Option<std::path::PathBuf>,

    #[structopt(name = "OUT", parse(from_os_str))]
    bal_out: std::path::PathBuf,
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

fn axis_angle_from_quaternion(q: &cgmath::Quaternion<f32>) -> Vector3<f32> {
    let q1 = q[1];
    let q2 = q[2];
    let q3 = q[3];

    let sin_theta = (q1 * q1 + q2 * q2 + q3 * q3).sqrt();
    let cos_theta = q[0];
    let two_theta = 2.0
        * (if cos_theta < 0.0 {
            f32::atan2(-sin_theta, -cos_theta)
        } else {
            f32::atan2(sin_theta, cos_theta)
        });
    let k = two_theta / sin_theta;
    Vector3::new(q1 * k, q2 * k, q3 * k)
}

// Generate camera locations by placing cameras on a regular grid throughout the image. Locations
// are then filtered based on their height.
// TODO: smarter pattern for points. choose initial point and expand search?
// TODO: keep generating points until we hit the target number
fn generate_cameras_grid(scene: &embree_rs::CommittedScene, num_points: usize) -> Vec<Camera> {
    let mut intersection_ctx = embree_rs::IntersectContext::coherent(); // not sure if this matters
    let mut positions = Vec::new();

    let poisson = poisson::Builder::<f32, na::Vector2<f32>>::with_samples(
        num_points * 2, // x2 seems to get us closer to the desired amount
        1.0,
        poisson::Type::Normal,
    )
    .build(rand::thread_rng(), poisson::algorithm::Ebeida);
    let samples = poisson.generate();

    let bounds = scene.bounds();
    // add a little wiggle room
    let start = Vector3::new(bounds.upper_x, bounds.upper_y, bounds.upper_z + 0.1);
    let delta = Vector3::new(
        bounds.upper_x - bounds.lower_x,
        bounds.upper_y - bounds.lower_y,
        0.0,
    );

    for sample in samples {
        let origin = start - delta.mul_element_wise(Vector3::new(sample[0], sample[1], 0.0));
        let direction = Vector3::new(0.0, 0.0, -1.0); // looking directly down
        let ray = embree_rs::Ray::new(origin, direction);
        let mut ray_hit = embree_rs::RayHit::new(ray);
        scene.intersect(&mut intersection_ctx, &mut ray_hit);
        if ray_hit.hit.hit() {
            // push point up a little from where it hit
            let pt = origin + direction * ray_hit.ray.tfar + Vector3::new(0.0, 0.0, 1.0);

            if pt[2] < 10.0 {
                positions.push(pt);
            }
        }
    }

    positions
        .into_iter()
        .map(|position| {
            // choose a random direction looking at the horizon
            // TODO: check that we are not too close to an object
            let dir = thread_rng().gen_range(0.0, 2.0 * std::f32::consts::PI);
            let down_x = cgmath::Quaternion::from_angle_y(cgmath::Rad(std::f32::consts::PI / 2.0));
            let around_z = cgmath::Quaternion::from_angle_z(cgmath::Rad(dir));
            Camera {
                loc: position,
                dir: axis_angle_from_quaternion(&(around_z * down_x)),
                intrin: Vector3::new(1.0, 0.0, 0.0),
                img_size: (1024, 1024),
            }
        })
        .collect::<Vec<_>>()
}

fn iter_triangles<'a>(
    model: &'a tobj::Model,
) -> impl Iterator<Item = (Vector3<f32>, Vector3<f32>, Vector3<f32>)> + 'a {
    model
        .mesh
        .indices
        .iter()
        .map(move |index| {
            let i: usize = *index as usize;
            Vector3::new(
                model.mesh.positions[i * 3],
                model.mesh.positions[i * 3 + 1],
                model.mesh.positions[i * 3 + 2],
            )
        })
        .tuples::<(_, _, _)>()
}

/// Choose a uniformly distributed random point in the given triangle.
fn random_point_in_triangle(v0: Vector3<f32>, v1: Vector3<f32>, v2: Vector3<f32>) -> Vector3<f32> {
    let mut rx = thread_rng().gen_range(0.0, 1.0);
    let mut ry = thread_rng().gen_range(0.0, 1.0);

    // sample from parallelogram, but reflect points outside of the original triangle
    if rx + ry > 1.0 {
        rx = 1.0 - rx;
        ry = 1.0 - ry;
    }

    v0 + rx * (v1 - v0) + ry * (v2 - v0)
}

fn generate_world_points_poisson(
    models: &Vec<tobj::Model>,
    num_points: usize,
) -> Vec<Vector3<f32>> {
    // calculate total area
    let total_area: f32 = models
        .iter()
        .map(|model| {
            iter_triangles(model)
                .map(|(v0, v1, v2)| ((v1 - v0).cross(v2 - v0).magnitude() / 2.0) as f32)
                .sum::<f32>()
        })
        .sum();

    let mut points = Vec::new();

    for model in models {
        for (v0, v1, v2) in iter_triangles(model) {
            let area = ((v1 - v0).cross(v2 - v0).magnitude() / 2.0) as f32;
            let mut num_samples = area / total_area * num_points as f32;
            // TODO: fix this probability
            while num_samples > 1.0 {
                points.push(random_point_in_triangle(v0, v1, v2));
                num_samples -= 1.0;
            }
            if thread_rng().gen_bool(num_samples as f64) {
                points.push(random_point_in_triangle(v0, v1, v2));
            }
        }
    }

    points
}

fn visibility_graph(
    scene: &embree_rs::CommittedScene,
    cameras: &Vec<Camera>,
    points: &Vec<Vector3<f32>>,
    max_dist: f32,
) -> Vec<Vec<(usize, (f32, f32))>> {
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
                if p_camera.z < 0.0 && (camera.loc - point).magnitude() < max_dist {
                    // check point is in front of camera
                    let p = camera.project(p_camera);

                    // check point is in camera frame
                    if p.x >= 0.0
                        && p.x < camera.img_size.0 as f32
                        && p.y >= 0.0
                        && p.y < camera.img_size.1 as f32
                    {
                        // ray pointing from camera towards the point
                        let dir = point - camera.loc;
                        let mut ray = embree_rs::Ray::new(camera.loc, dir.normalize());

                        // add rays to batch check
                        ray.tfar = dir.magnitude() - 1e-4;
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
    points: &Vec<Vector3<f32>>,
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
            point.insert("x".to_string(), Property::Float(camera.loc[0]));
            point.insert("y".to_string(), Property::Float(camera.loc[1]));
            point.insert("z".to_string(), Property::Float(camera.loc[2]));
            point.insert("red".to_string(), Property::UChar(255));
            point.insert("green".to_string(), Property::UChar(0));
            point.insert("blue".to_string(), Property::UChar(0));
            point
        })
        .collect();

    let pts = points.iter().map(|point| {
        let mut p = DefaultElement::new();
        p.insert("x".to_string(), Property::Float(point[0]));
        p.insert("y".to_string(), Property::Float(point[1]));
        p.insert("z".to_string(), Property::Float(point[2]));
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
    let (models_, _) = city_obj.unwrap();

    let models = normalize(&models_);

    // create embree device
    let dev = embree_rs::Device::new();

    let meshes = models
        .iter()
        .map(|model| add_model(model, &dev))
        .collect::<Vec<_>>();

    let mut scene = embree_rs::Scene::new(&dev);
    for mesh in meshes.into_iter() {
        scene.attach_geometry(mesh);
    }

    let cscene = scene.commit();

    let cameras = generate_cameras_grid(&cscene, opt.num_cameras);
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
    let bal_lcc = bal.largest_connected_component();
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
