//! Functions to generate camera and point locations on a 3D model.
//!
//! Example usage:
//! ```
//! extern crate tobj;
//! use city2ba::*;
//! use city2ba::generate::*;
//! use std::path::Path;
//!
//! // load the model from disk
//! let (models, _) = tobj::load_obj(Path::new("tests/box.obj")).expect("Could not load .obj");
//! // convert the model into a form used for fast intersection tests
//! let dev = embree_rs::Device::new();
//! let mut scene = embree_rs::Scene::new(&dev);
//! for model in models.iter() {
//!     let mesh = model_to_geometry(model, &dev);
//!     scene.attach_geometry(mesh);
//! }
//! let cscene = scene.commit();
//! // generate cameras
//! let cameras = generate_cameras_poisson::<SnavelyCamera>(&cscene, 100, 1., 0.);
//! // generate points
//! let points = generate_world_points_uniform(&models, &cameras, 200, 10.);
//! // compute camera-point visibility graph
//! let vis_graph = visibility_graph(&cscene, &cameras, &points, 10., false);
//! // create BA problem
//! let ba = BAProblem::from_visibility(cameras, points, vis_graph);
//! // drop all but the largest connected component and drop cameras that do no see enough points
//! let ba = ba.cull();
//! ```

extern crate cgmath;
extern crate embree_rs;
extern crate indicatif;
extern crate itertools;
extern crate nalgebra as na;
extern crate poisson;
extern crate rand;
extern crate rayon;
extern crate rstar;
extern crate structopt;
extern crate tobj;

use itertools::Itertools;
use rand::distributions::{Distribution, WeightedIndex};
use rand::{thread_rng, Rng};

use indicatif::{ParallelProgressIterator, ProgressBar, ProgressStyle};

use std::convert::TryInto;

use crate::baproblem::*;
use cgmath::prelude::*;
use cgmath::{Basis3, ElementWise, InnerSpace, Point3, Vector3, Vector4};
use rayon::prelude::*;
use rstar::RTree;

pub(crate) fn progress_bar(length: u64, message: &str, verbose: bool) -> ProgressBar {
    if !verbose {
        return ProgressBar::hidden();
    }

    let pb = ProgressBar::new(length);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{msg} [{bar:40}] {percent}% ({eta})")
            .progress_chars("#-"),
    );
    pb.set_message(message);
    pb
}

/// Convert a 3D model into geometry for fast intersection tests.
pub fn model_to_geometry<'a>(
    model: &tobj::Model,
    dev: &'a embree_rs::Device,
) -> embree_rs::Geometry<'a> {
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
pub fn generate_cameras_path<C: Camera>(
    _scene: &embree_rs::CommittedScene,
    path: &tobj::Model,
    num_cameras: usize,
) -> Vec<C> {
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

    // We sample randomly from the path by taking a weighted sample of each path segment (weighted
    // by segment length) and then uniformly sampling withing the segment.
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
            )
        })
        .collect()
}

/// Generate camera positions along a path by starting at the beginning of the path and taking
/// fixed sized steps between each camera.
pub fn generate_cameras_path_step<C>(
    _scene: &embree_rs::CommittedScene,
    path: &tobj::Model,
    num_cameras: usize,
    step_size: f64,
) -> Vec<C>
where
    C: Camera,
{
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
    let total_length = paths.iter().map(|(x, y)| (y - x).magnitude()).sum();
    assert!(
        num_cameras as f64 * step_size <= total_length,
        "Length of path {} is less than the number of cameras ({}) times the step size ({}) {}",
        total_length,
        num_cameras,
        step_size,
        num_cameras as f64 * step_size
    );

    println!(
        "Generating cameras along path. Path length: {}, using {} of it.",
        total_length,
        num_cameras as f64 * step_size
    );

    let mut segment_index = 0;
    let mut dist = 0.0;
    let mut cameras = Vec::with_capacity(num_cameras);
    for _ in 0..num_cameras {
        let (start, end) = paths[segment_index];
        let mut dir = end - start;
        let pos = start + dist / dir.magnitude() * dir;
        cameras.push(Camera::from_position_direction(
            pos,
            Basis3::between_vectors(dir.normalize(), Vector3::new(0.0, 0.0, -1.0)),
        ));

        // step to the next camera location
        dist += step_size;
        while dist >= dir.magnitude() {
            segment_index += 1;
            dist -= dir.magnitude();
            let (start, end) = paths[segment_index];
            dir = end - start;
        }
    }
    cameras
}

/// Generate camera locations using a Poisson disk distribution of cameras in the x-y plane.
/// Cameras are placed `height` above the tallest surface at their x-y location.
pub fn generate_cameras_poisson<C>(
    scene: &embree_rs::CommittedScene,
    num_points: usize,
    height: f64,
    ground: f64,
) -> Vec<C>
where
    C: Camera,
{
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
        bounds.upper_y as f64 + 0.1,
        bounds.upper_z as f64,
    );
    let delta = Vector3::new(
        (bounds.upper_x - bounds.lower_x) as f64,
        0.0,
        (bounds.upper_z - bounds.lower_z) as f64,
    );

    for sample in samples {
        let origin = start - delta.mul_element_wise(Vector3::new(sample[0], 0.0, sample[1]));
        let direction = Vector3::new(0.0, -1.0, 0.0); // looking directly down
        let ray = embree_rs::Ray::new(
            origin.cast::<f32>().unwrap().to_vec(),
            direction.cast::<f32>().unwrap(),
        );
        let mut ray_hit = embree_rs::RayHit::new(ray);
        scene.intersect(&mut intersection_ctx, &mut ray_hit);
        if ray_hit.hit.hit() {
            // push point up a little from where it hit
            let pt =
                origin + direction * (ray_hit.ray.tfar as f64) + Vector3::new(0.0, height, 0.0);

            if pt[2] < bounds.lower_y as f64 + ground {
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
            Camera::from_position_direction(position, around_z)
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
    let v = (0..3)
        .map(|j| {
            let i0 = mesh.indices[i * 3 + j] as usize;
            Point3::new(
                mesh.positions[i0 * 3],
                mesh.positions[i0 * 3 + 1],
                mesh.positions[i0 * 3 + 2],
            )
            .cast::<f64>()
            .unwrap()
        })
        .collect::<Vec<_>>();
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

#[derive(Debug, Clone, PartialEq, Copy)]
struct WrappedPoint(Point3<f64>);

impl rstar::Point for WrappedPoint {
    type Scalar = f64;
    const DIMENSIONS: usize = 3;

    fn generate(generator: impl Fn(usize) -> Self::Scalar) -> Self {
        WrappedPoint(Point3::new(generator(0), generator(1), generator(2)))
    }

    fn nth(&self, index: usize) -> Self::Scalar {
        let WrappedPoint(p) = self;
        match index {
            0 => p.x,
            1 => p.y,
            2 => p.z,
            _ => unreachable!(),
        }
    }

    fn nth_mut(&mut self, _index: usize) -> &mut Self::Scalar {
        unimplemented!()
    }
}

/// Generate points uniformly in the world. Points are culled if they are more than `max_dist` from
/// all cameras. Gives at most `num_points`.
pub fn generate_world_points_uniform<C>(
    models: &[tobj::Model],
    cameras: &[C],
    num_points: usize,
    max_dist: f64,
) -> Vec<Point3<f64>>
where
    C: Camera,
{
    if cameras.len() == 0 {
        panic!("Cannot generate world points with 0 cameras.");
    }

    // TODO: filter meshes by distance from cameras
    let areas = models.iter().flat_map(|model| {
        iter_triangles(&model.mesh)
            .map(|(v0, v1, v2)| ((v1 - v0).cross(v2 - v0).magnitude() / 2.0) as f64)
    });
    let indices = models
        .iter()
        .enumerate()
        .flat_map(|(i, model)| {
            iter_triangles(&model.mesh)
                .enumerate()
                .map(move |(j, _)| (i, j))
        })
        .collect::<Vec<_>>();

    let dist = WeightedIndex::new(areas).unwrap();
    let mut rng = thread_rng();

    let rtree = RTree::bulk_load(cameras.iter().map(|c| WrappedPoint(c.center())).collect());

    let mut points = Vec::with_capacity(num_points);
    let mut fail_count = 0;
    let fail_threshold = 10*num_points;
    while points.len() < num_points && fail_count < fail_threshold {
        let i = dist.sample(&mut rng);
        let (m, j) = indices[i];
        let mesh = &models[m].mesh;
        let (v0, v1, v2) = get_triangle(mesh, j);
        let p = random_point_in_triangle(v0, v1, v2);
        // check if point is close enough
        if rtree
            .locate_within_distance(WrappedPoint(p), max_dist * max_dist)
            .next()
            .is_some()
        {
            points.push(p);
        } else {
            fail_count += 1;
        }
    }

    if fail_count >= fail_threshold {
        panic!("Failed to generate enough points. {} successes, {} failures, {} requested points.", points.len(), fail_count, num_points);
    }

    points
}

/// Compute the camera-point visibility graph. Points not visible `max_dist` from any camera are
/// dropped.
pub fn visibility_graph<C>(
    scene: &embree_rs::CommittedScene,
    cameras: &[C],
    points: &[Point3<f64>],
    max_dist: f64,
    verbose: bool,
) -> Vec<Vec<(usize, (f64, f64))>>
where
    C: Camera + Sync,
{
    cameras
        .par_iter()
        .progress_with(progress_bar(
            cameras.len().try_into().unwrap(),
            "Computing Visibility",
            verbose,
        ))
        .map(|camera| {
            let mut intersection_ctx = embree_rs::IntersectContext::coherent(); // not sure if this matters

            let mut local_obs = Vec::with_capacity(points.len());
            let mut local_rays = Vec::with_capacity(points.len());
            for (i, point) in points.iter().enumerate() {
                // project point into camera frame
                let p_camera = camera.project_world(point);
                // check if point is in front of camera (camera looks down negative z)
                if (camera.center() - point).magnitude() < max_dist && p_camera.z <= 0.0 {
                    let p = camera.project(p_camera);

                    // check point is in camera frame
                    if p.x >= -1.0 && p.x <= 1.0 && p.y >= -1.0 && p.y <= 1.0 {
                        // ray pointing from camera towards the point
                        let dir = point - camera.center();
                        let mut ray = embree_rs::Ray::new(
                            camera.center().cast::<f32>().unwrap().to_vec(),
                            dir.normalize().cast::<f32>().unwrap(),
                        );

                        // Check if there is anything between the camera and the point. We stop a
                        // little short of the point to make sure we don't hit it.
                        ray.tfar = dir.magnitude() as f32 - 1e-6;
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
                .map(|x| x.1) // we just want the observation
                .collect()
        })
        .collect()
}

/// Move models so that the top right corner of the bounding box is at the origin.
pub fn move_to_origin(models: Vec<tobj::Model>) -> Vec<tobj::Model> {
    let min_vec = |x: Vector3<f32>, y: Vector3<f32>| {
        Vector3::new(x[0].min(y[0]), x[1].min(y[1]), x[2].min(y[2]))
    };
    let min_locs = &models
        .iter()
        .map(|model| {
            model
                .mesh
                .positions
                .chunks(3)
                .map(|chunk| Vector3::new(chunk[0], chunk[1], chunk[2]))
                .fold1(min_vec)
                .unwrap()
        })
        .fold1(min_vec)
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
                    vec![x[0], x[1], x[2]].into_iter()
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

/// Modify camera intrinsics so they are all in the range [`intrinsic_start`, `intrinsic_end`).
pub fn modify_intrinsics(
    cameras: &mut Vec<SnavelyCamera>,
    intrinsic_start: Vector3<f64>,
    intrinsic_end: Vector3<f64>,
) {
    let mut rng = rand::thread_rng();
    for camera in cameras.iter_mut() {
        let x = rng.gen_range(0.0, 1.0);
        let y = rng.gen_range(0.0, 1.0);
        let z = rng.gen_range(0.0, 1.0);
        let v = Vector3::new(x, y, z);
        camera.intrin =
            intrinsic_start + v.mul_element_wise(intrinsic_end.sub_element_wise(intrinsic_start));
    }
}
