extern crate cgmath;
extern crate embree;
extern crate itertools;
extern crate nalgebra as na;
extern crate petgraph;
extern crate ply_rs;
extern crate poisson;
extern crate rand;
extern crate rayon;
extern crate structopt;
extern crate tobj;

use itertools::Itertools;
use rand::{thread_rng, Rng};
use structopt::StructOpt;

use ply_rs::ply::{Addable, DefaultElement, ElementDef, Ply, Property, PropertyDef, PropertyType,
                  ScalarType};
use ply_rs::writer::Writer;

use std::fs::File;
use std::io::{BufWriter, Write};

use cgmath::prelude::*;
use cgmath::{ElementWise, InnerSpace, Vector3, Vector4};
use embree::Geometry;

use petgraph::visit::EdgeRef;

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

    #[structopt(long = "ply", parse(from_os_str))]
    ply_out: Option<std::path::PathBuf>,

    #[structopt(name = "OUT", parse(from_os_str))]
    bal_out: std::path::PathBuf,
}

fn add_model<'a>(model: &tobj::Model, dev: &'a embree::Device) -> embree::TriangleMesh<'a> {
    let num_tri = model.mesh.indices.len() / 3;
    let num_vert = model.mesh.positions.len() / 3;
    let mut mesh = embree::TriangleMesh::unanimated(dev, num_tri, num_vert);

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

    mesh.commit();
    mesh
}

fn axis_angle_from_quaternion(q: &cgmath::Quaternion<f32>) -> Vector3<f32> {
    let q1 = q[1];
    let q2 = q[2];
    let q3 = q[3];

    let sin_theta = (q1 * q1 + q2 * q2 + q3 * q3).sqrt();
    let cos_theta = q[0];
    let two_theta = 2.0 * (if cos_theta < 0.0 {
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
fn generate_cameras_grid(scene: &embree::Scene, num_points: usize) -> Vec<Camera> {
    let mut intersection_ctx = embree::IntersectContext::coherent(); // not sure if this matters
    let mut positions = Vec::new();

    let poisson = poisson::Builder::<f32, na::Vector2<f32>>::with_samples(
        num_points * 2, // x2 seems to get us closer to the desired amount
        1.0,
        poisson::Type::Normal,
    ).build(rand::thread_rng(), poisson::algorithm::Bridson);
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
        let ray = embree::Ray::new(origin, direction);
        let ray_hit = scene.intersect(&mut intersection_ctx, ray);
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
                focal_length: 1.0,
                distortion: (0.0, 0.0),
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
    let total_area: f64 = models
        .iter()
        .map(|model| {
            iter_triangles(model)
                .map(|(v0, v1, v2)| ((v1 - v0).cross(v2 - v0).magnitude() / 2.0) as f64)
                .sum::<f64>()
        })
        .sum();

    let mut points = Vec::new();

    for model in models {
        for (v0, v1, v2) in iter_triangles(model) {
            let area = ((v1 - v0).cross(v2 - v0).magnitude() / 2.0) as f64;
            let mut num_samples = area / total_area * num_points as f64;
            // TODO: fix this probability
            while num_samples > 1.0 {
                points.push(random_point_in_triangle(v0, v1, v2));
                num_samples -= 1.0;
            }
            if thread_rng().gen_bool(num_samples) {
                points.push(random_point_in_triangle(v0, v1, v2));
            }
        }
    }

    points
}

/*
fn generate_world_points(
    scene: &embree::Scene,
    cameras: &Vec<CameraPose>,
    num_points: usize,
) -> Vec<Vector3<f32>> {
    let img_size = (1024, 1024);

    let mut points = Vec::new();
    let mut intersection_ctx = embree::IntersectContext::incoherent(); // not sure if this matters

    for camera in cameras {
        let cam = Camera::look_dir(
            camera.loc,
            camera.dir,
            Vector3::new(0.0, 1.0, 0.0),
            75.0,
            img_size,
        );

        let poisson = poisson::Builder::<f32, na::Vector2<f32>>::with_samples(
            num_points / cameras.len() * 2, // x2 seems to get us closer to the desired amount
            1.0,
            poisson::Type::Normal,
        ).build(rand::thread_rng(), poisson::algorithm::Bridson);
        let samples = poisson.generate();

        for sample in samples {
            let dir = cam.ray_dir((sample[0] * img_size.0 as f32, sample[1] * img_size.0 as f32));
            let mut ray = embree::Ray::new(camera.loc, dir);
            ray.tnear = 0.001; // TODO: what to use here?
            let ray_hit = scene.intersect(&mut intersection_ctx, ray);
            if ray_hit.hit.hit() {
                points.push(camera.loc + ray_hit.ray.tfar * dir);
            }
        }
    }

    points
}
*/

fn visibility_graph(
    scene: &embree::Scene,
    cameras: &Vec<Camera>,
    points: &Vec<Vector3<f32>>,
) -> Vec<Vec<(usize, (f32, f32))>> {
    cameras
        .par_iter() // TODO: use par_iter
        .map(|camera| {
            let mut intersection_ctx = embree::IntersectContext::incoherent(); // not sure if this matters

            let mut local_obs = Vec::with_capacity(points.len());
            let mut local_rays = Vec::with_capacity(points.len());
            for (i, point) in points.iter().enumerate() {
                // project point into camera frame
                let p_camera = camera.project_world(point);
                // camera looks down negative z
                if p_camera.z < 0.0 {
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
                        let mut ray = embree::Ray::new(camera.loc, dir.normalize());

                        // add rays to batch check
                        ray.tfar = dir.magnitude() - 1e-4;
                        local_rays.push(ray);
                        local_obs.push((i, (p.x, p.y)));
                    }
                }
            }

            // filter by occluded rays
            scene
                .occluded_vec(&mut intersection_ctx, local_rays)
                .iter()
                .zip(local_obs)
                .filter(|x| !*x.0)
                .map(|x| x.1)
                .collect()
        }).collect()
}

/// Get the largest connected component of cameras and points.
fn largest_connected_component(
    vis_graph: &Vec<Vec<(usize, (f32, f32))>>,
    cameras: &Vec<Camera>,
    points: &Vec<Vector3<f32>>,
) -> (
    Vec<Vec<(usize, (f32, f32))>>,
    Vec<Camera>,
    Vec<Vector3<f32>>,
) {
    let num_edges = vis_graph
        .iter()
        .map(|adj| adj.iter().map(|x| x.0).sum::<usize>())
        .sum();
    let num_cameras = vis_graph.len();
    let mut g = petgraph::Graph::<bool, _, petgraph::Undirected, usize>::with_capacity(
        num_cameras,
        num_edges,
    );
    for (i, adj) in vis_graph.iter().enumerate() {
        g.extend_with_edges(adj.iter().map(|o| (i, o.0 + num_cameras, o.1)));
    }

    let cc = petgraph::algo::tarjan_scc(&g);
    let largest = cc.iter().max_by_key(|&v| v.len()).unwrap();

    let mut node_id_map = vec![None; g.node_count()];
    let num_cameras_left = largest.iter().filter(|i| i.index() < num_cameras).count();
    let mut camera_count = 0;
    let mut point_count = 0;
    let mut new_points = Vec::new();
    let mut new_cameras = Vec::new();

    for node_id in largest {
        if node_id.index() >= num_cameras {
            // is a point
            node_id_map[node_id.index()] = Some(num_cameras_left + point_count);
            new_points.push(points[node_id.index() - num_cameras]);
            point_count += 1;
        } else {
            // is a camera
            node_id_map[node_id.index()] = Some(camera_count);
            new_cameras.push(cameras[node_id.index()].clone());
            camera_count += 1;
        }
    }

    // store node ids in the weight of the node
    let lcc = g.filter_map(
        |node_id, _| node_id_map[node_id.index()],
        |_, edge| Some(edge),
    );

    let mut adj = vec![Vec::new(); num_cameras_left];
    for edge in lcc.edge_references() {
        let (c, p) = if edge.source().index() < num_cameras_left {
            (edge.source(), edge.target())
        } else {
            (edge.target(), edge.source())
        };

        adj[*lcc.node_weight(c).unwrap()].push((
            lcc.node_weight(p).unwrap() - num_cameras_left,
            **edge.weight(),
        ));
    }

    (adj, new_cameras, new_points)
}

fn write_bal(
    path: &std::path::Path,
    cameras: &Vec<Camera>,
    points: &Vec<Vector3<f32>>,
    vis_graph: &Vec<Vec<(usize, (f32, f32))>>,
) {
    let mut file = BufWriter::new(File::create(path).unwrap());
    writeln!(
        &mut file,
        "{} {} {}",
        cameras.len(),
        points.len(),
        vis_graph.iter().map(|x| x.len()).sum::<usize>()
    );
    for (i, obs) in vis_graph.iter().enumerate() {
        for (p, (u, v)) in obs {
            writeln!(&mut file, "{} {} {} {}", i, p, u, v);
        }
    }

    // TODO: actually pass around the intrinsics
    for camera in cameras {
        writeln!(
            &mut file,
            "{} {} {} {} {} {} {} {} {}",
            camera.dir[0],
            camera.dir[1],
            camera.dir[2],
            camera.loc[0],
            camera.loc[1],
            camera.loc[2],
            1.0,
            0.0,
            0.0
        );
    }

    for point in points {
        writeln!(&mut file, "{} {} {}", point[0], point[1], point[2]);
    }
}

/// Write camera locations out to a ply file. Does not provide color or orientation.
fn write_cameras(path: &std::path::Path, cameras: &Vec<Camera>, points: &Vec<Vector3<f32>>) {
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

    let mut file = BufWriter::new(File::create(path).unwrap());
    let writer = Writer::new();
    writer.write_ply(&mut file, &mut ply).unwrap();
}

fn main() {
    let opt = Opt::from_args();

    let city_obj = tobj::load_obj(&opt.input);
    let (models, _) = city_obj.unwrap();

    // create embree device
    let dev = embree::Device::new();

    let meshes = models
        .iter()
        .map(|model| add_model(model, &dev))
        .collect::<Vec<_>>();

    let mut scene = embree::Scene::new(&dev);
    for mesh in meshes.iter() {
        scene.attach_geometry(mesh);
    }

    scene.commit();

    let cameras = generate_cameras_grid(&scene, opt.num_cameras);
    println!("Generated {} cameras", cameras.len());

    let points = generate_world_points_poisson(&models, opt.num_world_points);
    println!("Generated {} world points", points.len());

    let vis_graph = visibility_graph(&scene, &cameras, &points);
    println!(
        "Computed visibility graph with {} edges",
        vis_graph.iter().map(|x| x.len()).sum::<usize>()
    );
    let (lcc, lcc_cameras, lcc_points) = largest_connected_component(&vis_graph, &cameras, &points);
    println!(
        "Computed LCC with {} cameras, {} points, {} edges",
        lcc_cameras.len(),
        lcc_points.len(),
        lcc.iter().map(|x| x.len()).sum::<usize>()
    );

    println!(
        "Total reprojection error: {}",
        total_reprojection_error(&lcc, &lcc_cameras, &lcc_points)
    );

    write_bal(&opt.bal_out, &lcc_cameras, &lcc_points, &lcc);

    match opt.ply_out {
        Some(path) => write_cameras(&path, &lcc_cameras, &lcc_points),
        None => (),
    }
}
