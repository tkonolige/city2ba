extern crate cgmath;
extern crate geo;
extern crate indicatif;
extern crate line_intersection;
extern crate rayon;
extern crate rstar;

use cgmath::prelude::*;
use cgmath::{Basis3, Point3, Vector3};
use geo::Line;
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressStyle};
use line_intersection::LineInterval;
use rayon::prelude::*;
use rstar::RTree;
use std::convert::TryInto;

use crate::baproblem::*;

#[derive(Debug, Clone, PartialEq, Copy)]
pub struct WrappedPoint(Point3<f64>, usize);

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
            let side_segment = LineInterval::line_segment(Line {
                start: side.0.into(),
                end: side.1.into(),
            });
            let view_segment = LineInterval::line_segment(Line {
                start: start.into(),
                end: end.into(),
            });
            // println!("{:?}", side);

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

/// Generate a synthetic scene of buildings on a grid
///
///   +-------+ +-------+
///  * * * * * * * * * * *
/// +*+-------+*+-------+*+
/// |*|       |*|       |*|
/// +*+-------+*+-------+*+
///  * * * * * * * * * * *
///   +-------+ +-------+
pub fn synthetic_grid(
    num_cameras_per_block: usize,
    num_points_per_block: usize,
    block_inset: f64,
    num_blocks: usize,
    block_length: f64,
    camera_height: f64,
    point_height: f64,
    max_dist: f64,
) -> BAProblem {
    assert!(block_inset * 2. < block_length);
    let mut cameras = Vec::new();
    for bx in 0..num_blocks {
        let offset_x = block_length * bx as f64;
        for by in 0..num_blocks {
            let offset_z = block_length * by as f64;
            for i in 0..num_cameras_per_block {
                // horizontal block
                let loc_x = Point3::new(
                    offset_x + i as f64 / num_cameras_per_block as f64 * block_length,
                    camera_height,
                    offset_z,
                );
                let dir_x = Basis3::from_angle_y(cgmath::Deg(-90.));
                cameras.push(Camera::from_position_direction(
                    loc_x,
                    dir_x,
                    Vector3::new(1.0, 0.0, 0.0),
                ));
                // vertical block
                let loc_z = Point3::new(
                    offset_x,
                    camera_height,
                    offset_z + i as f64 / num_cameras_per_block as f64 * block_length,
                );
                let dir_z = Basis3::from_angle_z(cgmath::Deg(180.));
                cameras.push(Camera::from_position_direction(
                    loc_z,
                    dir_z,
                    Vector3::new(1.0, 0.0, 0.0),
                ));
            }
        }
    }

    // Points need to be on each side of the cameras
    let mut points = Vec::new();
    for bx in 0..num_blocks {
        let offset_x = block_length * bx as f64;
        for by in 0..num_blocks {
            let offset_z = block_length * by as f64;
            for i in 0..num_points_per_block {
                // horizontal block, both left and right
                let loc_x = offset_x
                    + block_inset
                    + i as f64 / num_points_per_block as f64 * (block_length - block_inset * 2.);
                points.push(Point3::new(loc_x, point_height, offset_z - block_inset));
                points.push(Point3::new(loc_x, point_height, offset_z + block_inset));
                // vertical block, both left and right
                let loc_z = offset_z
                    + block_inset
                    + i as f64 / num_points_per_block as f64 * (block_length - block_inset * 2.);
                points.push(Point3::new(offset_x - block_inset, point_height, loc_z));
                points.push(Point3::new(offset_x + block_inset, point_height, loc_z));
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
    let pb = ProgressBar::new(cameras.len().try_into().unwrap());
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{bar:40}] {percent}% ({eta})")
            .progress_chars("#-"),
    );
    let visibility = cameras
        .par_iter()
        .progress_with(pb)
        .map(|camera| {
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

    BAProblem::from_visibility(cameras, points, visibility).cull()
}
