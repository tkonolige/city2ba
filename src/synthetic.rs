//! Functions for generating cameras and points without reference geometry.

extern crate cgmath;
extern crate geo;
extern crate indicatif;
extern crate line_intersection;
extern crate rayon;
extern crate rstar;

use cgmath::prelude::*;
use cgmath::{Basis3, Point3};
use indicatif::ParallelProgressIterator;
use line_intersection::LineInterval;
use rayon::prelude::*;
use rstar::RTree;
use std::convert::TryInto;

use crate::baproblem::*;
use crate::generate::progress_bar;

#[derive(Debug, Clone, PartialEq, Copy)]
struct WrappedPoint(Point3<f64>, usize);

impl rstar::Point for WrappedPoint {
    type Scalar = f64;
    const DIMENSIONS: usize = 3;

    fn generate(generator: impl Fn(usize) -> Self::Scalar) -> Self {
        WrappedPoint(
            Point3::new(generator(0), generator(1), generator(2)),
            std::usize::MAX,
        )
    }

    fn nth(&self, index: usize) -> Self::Scalar {
        let WrappedPoint(p, _) = self;
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

/// Checks if the line segment defined by (`start`, `end`) hits any building side in the block. Returns
/// false if the intersection occurs at `end`.
fn hits_in_block(
    start: (f64, f64),
    end: (f64, f64),
    block_index: (isize, isize),
    block_length: f64,
    block_inset: f64,
) -> bool {
    let block_end = block_length - block_inset;
    let offset_x = block_index.0 as f64 * block_length;
    let offset_y = block_index.1 as f64 * block_length;
    let sides = [
        (
            (offset_x + block_inset, offset_y + block_inset),
            (offset_x + block_inset, offset_y + block_end),
        ),
        (
            (offset_x + block_inset, offset_y + block_inset),
            (offset_x + block_end, offset_y + block_inset),
        ),
        (
            (offset_x + block_end, offset_y + block_inset),
            (offset_x + block_end, offset_y + block_end),
        ),
        (
            (offset_x + block_inset, offset_y + block_end),
            (offset_x + block_end, offset_y + block_end),
        ),
    ];
    sides
        .iter()
        .map(|side| {
            let side_segment = LineInterval::line_segment(geo::Line {
                start: side.0.into(),
                end: side.1.into(),
            });
            let view_segment = LineInterval::line_segment(geo::Line {
                start: start.into(),
                end: end.into(),
            });
            match view_segment.relate(&side_segment).unique_intersection() {
                // make sure we didn't hit the end point
                Some(p) => ((end.0 - p.x()).powf(2.) + (end.1 - p.y())).sqrt() > 1e-8,
                None => false,
            }
        })
        .any(|x| x)
}

fn hits_building(c: Point3<f64>, p: Point3<f64>, block_length: f64, block_inset: f64) -> bool {
    // returns (block x, block z)
    let block_index = |x: (f64, f64)| {
        (
            (x.0 / block_length).trunc() as isize,
            (x.1 / block_length).trunc() as isize,
        )
    };

    let start = (c.x, c.z);
    let end = (p.x, p.z);
    let (cbx, cby) = block_index(start);
    let (pbx, pby) = block_index(end);
    // iterate over all block in quadrant between camera and point
    // TODO: could simplify by only hitting blocks on line between camera and point
    // Using inclusive ranges to hit both start and end blocks
    (match (cbx < pbx, cby < pby) {
        (true, true) => itertools::iproduct!(cbx..=pbx, cby..=pby),
        (true, false) => itertools::iproduct!(cbx..=pbx, pby..=cby),
        (false, true) => itertools::iproduct!(pbx..=cbx, cby..=pby),
        (false, false) => itertools::iproduct!(pbx..=cbx, pby..=cby),
    })
    .map(|i| hits_in_block(start, end, i, block_length, block_inset))
    .any(|x| x)
}

/// Generate a synthetic scene of buildings on a 2D grid.
///
/// ```txt
///  num_cameras_per_block
///  ----------------
///  * * * * * * * * * * * * * * *   |
///  *               *               | block inset
///  *  +---------+  *  +---------+  |
///  *  |         |  *  |         |
///  *  |         |  *  |         |
///  *  |         |  *  |         |
///  *  |         |  *  |         |
///  *  +---------+  *  +---------+
///  *               *
///  * * * * * * * * * * * * * * *
///  *               *
///  *  +---------+  *  +---------+
///  *  |         |  *  |         |
///  *  |         |  *  |         |
///  *  |         |  *  |         |
///  *  |         |  *  |         |
///  *  +---------+  *  +---------+
///                     -----------
///                     points_per_block
///  ```
///
/// ## Arguments
/// - `num_cameras_per_block`: number of cameras per grid block.
/// - `num_points_per_block`: number of points per grid block.
/// - `num_blocks`: grid is `num_blocks` by `num_blocks`.
/// - `block_length`: the length of each block.
/// - `block_inset`: offset between cameras and points on a block.
/// - `camera_height`: height of cameras from the ground (in y).
/// - `point_height`: height of points from the ground (in y). Points are also added at y=0 at
///    block_inset/2.
/// - `max_dist`: maximum distance between a camera and a point for visibility.
/// - `verbose`: should a progress bar be displayed.
pub fn synthetic_grid<C>(
    num_cameras_per_block: usize,
    num_points_per_block: usize,
    num_blocks: usize,
    block_length: f64,
    block_inset: f64,
    camera_height: f64,
    point_height: f64,
    max_dist: f64,
    verbose: bool,
) -> BAProblem<C>
where
    C: Camera + Sync + Clone,
{
    assert!(block_inset * 2. < block_length, "Block inset ({}) must be less than half the block length ({}), to not violate physical constraints.", block_inset, block_length);
    let mut cameras = Vec::new();
    for bx in 0..=num_blocks {
        let offset_x = block_length * bx as f64;
        for by in 0..=num_blocks {
            let offset_z = block_length * by as f64;
            for i in 0..num_cameras_per_block {
                // horizontal block
                if bx != num_blocks {
                    let loc_x = Point3::new(
                        offset_x + i as f64 / num_cameras_per_block as f64 * block_length,
                        camera_height,
                        offset_z,
                    );
                    let dir_x = Basis3::from_angle_y(cgmath::Deg(-90.));
                    cameras.push(Camera::from_position_direction(loc_x, dir_x));
                    let dir_x = Basis3::from_angle_y(cgmath::Deg(90.));
                    cameras.push(Camera::from_position_direction(loc_x, dir_x));
                }
                // vertical block
                if by != num_blocks {
                    let loc_z = Point3::new(
                        offset_x,
                        camera_height,
                        offset_z + i as f64 / num_cameras_per_block as f64 * block_length,
                    );
                    let dir_z = Basis3::from_angle_y(cgmath::Deg(180.));
                    cameras.push(Camera::from_position_direction(loc_z, dir_z));
                    let dir_z = Basis3::one();
                    cameras.push(Camera::from_position_direction(loc_z, dir_z));
                }
            }
        }
    }

    // Points need to be on each side of the cameras and below them
    let mut points = Vec::new();
    for bx in 0..=num_blocks {
        let offset_x = block_length * bx as f64;
        for by in 0..=num_blocks {
            let offset_z = block_length * by as f64;
            for i in 0..num_points_per_block {
                let step = (block_length - block_inset * 2.) / num_points_per_block as f64;
                // horizontal block, both left and right
                if bx != num_blocks {
                    let loc_x = offset_x + block_inset + i as f64 * step;
                    points.push(Point3::new(loc_x, point_height, offset_z - block_inset));
                    points.push(Point3::new(loc_x, point_height, offset_z + block_inset));
                    points.push(Point3::new(loc_x + step / 2., 0., offset_z - block_inset));
                    points.push(Point3::new(loc_x + step / 2., 0., offset_z + block_inset));
                    points.push(Point3::new(
                        loc_x + step / 2.,
                        0.,
                        offset_z - block_inset / 2.,
                    ));
                    points.push(Point3::new(
                        loc_x + step / 2.,
                        0.,
                        offset_z + block_inset / 2.,
                    ));
                }
                // vertical block, both left and right
                if by != num_blocks {
                    let loc_z = offset_z + block_inset + i as f64 * step;
                    points.push(Point3::new(offset_x - block_inset, point_height, loc_z));
                    points.push(Point3::new(offset_x + block_inset, point_height, loc_z));
                    points.push(Point3::new(offset_x - block_inset, 0., loc_z + step / 2.));
                    points.push(Point3::new(offset_x + block_inset, 0., loc_z + step / 2.));
                    points.push(Point3::new(
                        offset_x - block_inset / 2.,
                        0.,
                        loc_z + step / 2.,
                    ));
                    points.push(Point3::new(
                        offset_x + block_inset / 2.,
                        0.,
                        loc_z + step / 2.,
                    ));
                }
            }
        }
    }

    // compute visibility
    let rtree = RTree::bulk_load(
        points
            .iter()
            .enumerate()
            .map(|(i, p)| WrappedPoint(*p, i))
            .collect(),
    );
    let visibility = cameras
        .par_iter()
        .progress_with(progress_bar(
            cameras.len().try_into().unwrap(),
            "Computing visibility",
            verbose,
        ))
        .map(|camera: &C| {
            let mut obs = Vec::new();
            for p in rtree.locate_within_distance(
                WrappedPoint(camera.center(), std::usize::MAX),
                max_dist * max_dist,
            ) {
                let WrappedPoint(point, pi) = p;

                // check if point crosses any building
                if !hits_building(camera.center(), *point, block_length, block_inset) {
                    let p_camera = camera.project_world(point);
                    // camera looks down -z
                    if (camera.center() - point).magnitude() < max_dist && p_camera.z <= 0. {
                        let p = camera.project(p_camera);
                        if p.x >= -1. && p.x <= 1. && p.y >= -1. && p.y <= 1. {
                            obs.push((*pi, (p.x, p.y)));
                        }
                    }
                }
            }
            obs
        })
        .collect();

    BAProblem::from_visibility(cameras, points, visibility) // .cull()
}

/// Generate a series of synthetic cameras in a line.
///
/// ## Arguments
/// - `num_cameras`: number of cameras to generate. Final number of cameras might be slightly less.
/// - `num_points`: number of points to generate. Final number of points might be slightly less.
/// - `length`: length of the line the cameras are placed on.
/// - `point_offset`: points are placed this distance to the right and left of the cameras.
/// - `camera_height`: height of cameras from the ground (in y).
/// - `point_height`: height of points from the ground (in y).
/// - `max_dist`: maximum distance between a camera and a point for visibility.
/// - `verbose`: should a progress bar be displayed.
pub fn synthetic_line<C: Camera + Sync + Clone>(
    num_cameras: usize,
    num_points: usize,
    length: f64,
    point_offset: f64,
    camera_height: f64,
    point_height: f64,
    max_dist: f64,
    verbose: bool,
) -> BAProblem<C> {
    let cameras = (0..num_cameras)
        .map(|i| {
            let loc = Point3::new(
                0.,
                camera_height,
                i as f64 * length / (num_cameras - 1) as f64,
            );
            let dir = Basis3::from_angle_y(cgmath::Deg(180.));
            Camera::from_position_direction(loc, dir)
        })
        .collect::<Vec<_>>();
    let points = (0..num_points)
        .map(|i| {
            let z = (i / 2) as f64 * length / (num_points / 2 - 1) as f64;
            let x = if i % 2 == 0 {
                -point_offset
            } else {
                point_offset
            };
            Point3::new(x, point_height, z)
        })
        .collect::<Vec<_>>();

    let rtree = RTree::bulk_load(
        points
            .iter()
            .enumerate()
            .map(|(i, p)| WrappedPoint(*p, i))
            .collect(),
    );
    let visibility = cameras
        .par_iter()
        .progress_with(progress_bar(
            cameras.len().try_into().unwrap(),
            "Computing visibility",
            verbose,
        ))
        .map(|camera: &C| {
            let mut obs = Vec::new();
            for p in rtree.locate_within_distance(
                WrappedPoint(camera.center(), std::usize::MAX),
                max_dist * max_dist,
            ) {
                let WrappedPoint(point, pi) = p;

                let p_camera = camera.project_world(point);
                // camera looks down -z
                if (camera.center() - point).magnitude() < max_dist && p_camera.z <= 0. {
                    let p = camera.project(p_camera);
                    if p.x >= -1. && p.x <= 1. && p.y >= -1. && p.y <= 1. {
                        obs.push((*pi, (p.x, p.y)));
                    }
                }
            }
            obs
        })
        .collect();
    BAProblem::from_visibility(cameras, points, visibility).cull()
}
