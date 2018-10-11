extern crate rayon;
extern crate cgmath;
extern crate nom;

use rayon::prelude::*;
use cgmath::{InnerSpace, Vector3};
use nom::*;

use std::path::Path;
use std::io::prelude::*;
use std::fs::File;
use std::str::FromStr;

#[derive(Debug, Clone)]
pub struct Camera {
    pub loc: Vector3<f32>,
    pub dir: Vector3<f32>, // Rodriguez
    pub focal_length: f32,
    pub distortion: (f32, f32),
    pub img_size: (usize, usize),
}

impl Camera {
    /// Project a point from the world into the camera coordinate system
    pub fn project_world(&self, p: &Vector3<f32>) -> cgmath::Vector3<f32> {
        let angle = cgmath::Rad(self.dir.magnitude());
        let axis = self.dir.normalize();
        cgmath::Matrix3::from_axis_angle(axis, angle) * p + self.loc
    }

    /// Project a point from camera space into pixel coordinates
    pub fn project(&self, p: cgmath::Vector3<f32>) -> cgmath::Vector2<f32> {
        let p_ = cgmath::Vector2::new(-p.x / p.z, -p.y / p.z);
        let r = 1.0 + self.distortion.0 * p_.magnitude2()
            + self.distortion.1 * p_.magnitude().powf(4.0);
        self.focal_length * r * p_
    }

    pub fn from_vec(x: Vec<f32>) -> Self {
        Camera{ dir: Vector3::new(x[0], x[1], x[2]), loc: Vector3::new(x[3], x[4], x[5]), focal_length: x[6], distortion: (x[7], x[8]), img_size: (1024, 1024)}
    }
}

pub fn total_reprojection_error(
    vis_graph: &Vec<Vec<(usize, (f32, f32))>>,
    cameras: &Vec<Camera>,
    points: &Vec<Vector3<f32>>,
) -> f32 {
    cameras
        .par_iter()
        .zip(vis_graph)
        .map(|(camera, adj)| {
            adj.iter()
                .map(|(o, (u, v))| {
                    let p = camera.project(camera.project_world(&points[*o]));
                    (p.x - u).abs() + (p.y - v).abs()
                })
                .sum::<f32>()
        })
        .sum()
}

#[derive(Debug)]
pub struct BALProblem {
    cameras: Vec<Camera>,
    points: Vec<Vector3<f32>>,
    vis_graph: Vec<Vec<(usize, (f32, f32))>>
}

impl BALProblem {
    pub fn total_reprojection_error(&self) -> f32 {
        total_reprojection_error(&self.vis_graph, &self.cameras, &self.points)
    }

    pub fn new(cams: Vec<Camera>, points: Vec<Vector3<f32>>, obs: Vec<(usize, usize, f32, f32)>) -> Self {
        let mut vis_graph = vec![Vec::new(); cams.len()];
        for (cam_i, p_i, obs_x, obs_y) in obs {
            vis_graph[cam_i].push((p_i, (obs_x, obs_y)));
        }

        BALProblem{cameras:cams, points:points, vis_graph:vis_graph}
    }

    pub fn from_file(filepath: &Path) -> BALProblem {
        named!(integer<&[u8], usize>, map_res!(map_res!(digit1, std::str::from_utf8), usize::from_str));

        named!(header<&[u8], (usize, usize, usize)>, ws!(tuple!(integer, integer, integer)));

        named!(observation<&[u8], (usize, usize, f32, f32)>, do_parse!(c_i: integer >> space1 >> p_i : integer >> space1 >> obs_x: float >> space1 >> obs_y: float >> ((c_i, p_i, obs_x, obs_y))));

        fn from_vec(x: Vec<f32>) -> Result<Camera, u8> {
            Ok(Camera::from_vec(x))
        }

        named!(camera<&[u8], Camera>, map_res!(count!(preceded!(space0, float), 9), from_vec));

        named!(point<&[u8], Vector3<f32> >, do_parse!(x: float >> space1 >> y: float >> space1 >> z: float >> (Vector3::new(x, y, z))));

        named!(bal_problem<&[u8], BALProblem>,
            do_parse!(hdr: header >>
                      obs: count!(preceded!(opt!(tag!("\n")), observation), hdr.2) >>
                      cams: count!(preceded!(tag!("\n"), camera), hdr.0) >>
                      pts: count!(preceded!(tag!("\n"), point), hdr.1) >>
                      (BALProblem::new(cams, pts, obs))
                  ));

        let mut file = File::open(filepath).expect("Could not open file");
        let mut contents = Vec::new();
        file.read_to_end(&mut contents).expect("Could not read file");
        bal_problem(contents.as_slice()).expect("Could not parse file").1
    }
}


