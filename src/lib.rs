extern crate cgmath;
extern crate nom;
extern crate rayon;
extern crate petgraph;

use cgmath::{ElementWise, InnerSpace, Vector3};
use nom::*;
use rayon::prelude::*;
use petgraph::visit::EdgeRef;

use std::fs::File;
use std::io::prelude::*;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::str::FromStr;

#[derive(Debug)]
pub enum Error {
    ParseError,
    IOError(std::io::Error)
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::IOError(e)
    }
}

impl<I> From<nom::Err<I>> for Error {
    fn from(_: nom::Err<I>) -> Self {
        Error::ParseError
    }
}

#[derive(Debug, Clone)]
pub struct Camera {
    pub loc: Vector3<f32>,
    pub dir: Vector3<f32>, // Rodriguez
    pub intrin: Vector3<f32>, // Rodriguez
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
        let r = 1.0
            + self.distortion().0 * p_.magnitude2()
            + self.distortion().1 * p_.magnitude().powf(4.0);
        self.focal_length() * r * p_
    }

    pub fn from_vec(x: Vec<f32>) -> Self {
        Camera {
            dir: Vector3::new(x[0], x[1], x[2]),
            loc: Vector3::new(x[3], x[4], x[5]),
            intrin: Vector3::new(x[6], x[7], x[8]),
            img_size: (1024, 1024),
        }
    }

    pub fn focal_length(&self) -> f32 {
        self.intrin[0]
    }

    pub fn distortion(&self) -> (f32, f32) {
        (self.intrin[1], self.intrin[2])
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

pub fn total_reprojection_error_l2(
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
                    (p.x - u).powf(2.0) + (p.y - v).powf(2.0)
                })
                .sum::<f32>()
        })
        .sum::<f32>()
        .sqrt()
        * 2.0
}

#[derive(Debug)]
pub struct BALProblem {
    pub cameras: Vec<Camera>,
    pub points: Vec<Vector3<f32>>,
    pub vis_graph: Vec<Vec<(usize, (f32, f32))>>,
}

impl BALProblem {
    pub fn total_reprojection_error(&self) -> f32 {
        total_reprojection_error(&self.vis_graph, &self.cameras, &self.points)
    }

    pub fn total_reprojection_error_l2(&self) -> f32 {
        total_reprojection_error_l2(&self.vis_graph, &self.cameras, &self.points)
    }

    pub fn mean(&self) -> Vector3<f32> {
        let num = (self.cameras.len() + self.points.len()) as f32;
        self.cameras
            .iter()
            .map(|x| &x.loc)
            .chain(self.points.iter())
            .fold(Vector3::new(0.0, 0.0, 0.0), |a, &b| a + b / num)
    }

    pub fn std(&self) -> Vector3<f32> {
        let num = (self.cameras.len() + self.points.len()) as f32;
        let mean = self.mean();
        (self
            .cameras
            .iter()
            .map(|x| &x.loc)
            .chain(self.points.iter())
            .map(|x| (x - mean).mul_element_wise(x - mean))
            .sum::<Vector3<f32>>()
            / num)
            .map(|x| x.sqrt())
    }

    pub fn new(
        cams: Vec<Camera>,
        points: Vec<Vector3<f32>>,
        obs: Vec<(usize, usize, f32, f32)>,
    ) -> Self {
        let mut vis_graph = vec![Vec::new(); cams.len()];
        for (cam_i, p_i, obs_x, obs_y) in obs {
            vis_graph[cam_i].push((p_i, (obs_x, obs_y)));
        }

        BALProblem {
            cameras: cams,
            points: points,
            vis_graph: vis_graph,
        }
    }

    pub fn from_file(filepath: &Path) -> Result<BALProblem, Error> {
        named!(integer<&[u8], usize>,
               map_res!(map_res!(digit1, std::str::from_utf8), usize::from_str));

        named!(header<&[u8], (usize, usize, usize)>, ws!(tuple!(integer, integer, integer)));

        named!(observation<&[u8], (usize, usize, f32, f32)>,
        do_parse!(c_i: integer >>
                  space1 >>
                  p_i : integer >>
                  space1 >>
                  obs_x: float >>
                  space1 >>
                  obs_y: float >>
                  ((c_i, p_i, obs_x, obs_y))
                  ));

        fn from_vec(x: Vec<f32>) -> Result<Camera, u8> {
            Ok(Camera::from_vec(x))
        }

        named!(camera<&[u8], Camera>,
               map_res!(count!(preceded!(space0, float), 9), from_vec));

        named!(point<&[u8], Vector3<f32> >,
        do_parse!(x: float >>
                  space1 >>
                  y: float >>
                  space1 >>
                  z: float >>
                  (Vector3::new(x, y, z))
                  ));

        named!(bal_problem<&[u8], BALProblem>,
        do_parse!(hdr: header >>
                  obs: count!(preceded!(opt!(tag!("\n")), observation), hdr.2) >>
                  cams: count!(dbg_dmp!(preceded!(tag!("\n"), camera)), hdr.0) >>
                  pts: dbg_dmp!(count!(preceded!(tag!("\n"), point), hdr.1)) >>
                  (BALProblem::new(cams, pts, obs))
              ));

        let mut file = File::open(filepath)?;
        let mut contents = Vec::new();
        file.read_to_end(&mut contents)?;
        bal_problem(contents.as_slice()).map(|x| x.1).map_err(Error::from)
    }

    /// Get the largest connected component of cameras and points.
    pub fn largest_connected_component(&self) -> Self {
        let num_edges = self.vis_graph
            .iter()
            .map(|adj| adj.iter().map(|x| x.0).sum::<usize>())
            .sum();
        let num_cameras = self.vis_graph.len();
        let mut g = petgraph::Graph::<bool, _, petgraph::Undirected, usize>::with_capacity(
            num_cameras,
            num_edges,
        );
        for (i, adj) in self.vis_graph.iter().enumerate() {
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
                new_points.push(self.points[node_id.index() - num_cameras]);
                point_count += 1;
            } else {
                // is a camera
                node_id_map[node_id.index()] = Some(camera_count);
                new_cameras.push(self.cameras[node_id.index()].clone());
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

        BALProblem{cameras:new_cameras, points:new_points, vis_graph:adj}
    }

    pub fn write(&self, path: &std::path::Path) -> Result<(), std::io::Error> {
        let mut file = BufWriter::new(File::create(path).unwrap());
        writeln!(
            &mut file,
            "{} {} {}",
            self.cameras.len(),
            self.points.len(),
            self.vis_graph.iter().map(|x| x.len()).sum::<usize>()
        )?;
        for (i, obs) in self.vis_graph.iter().enumerate() {
            for (p, (u, v)) in obs {
                writeln!(&mut file, "{} {} {} {}", i, p, u, v)?;
            }
        }

        for camera in &self.cameras {
            writeln!(
                &mut file,
                "{} {} {} {} {} {} {} {} {}",
                camera.dir[0],
                camera.dir[1],
                camera.dir[2],
                camera.loc[0],
                camera.loc[1],
                camera.loc[2],
                camera.intrin[0],
                camera.intrin[1],
                camera.intrin[2],
            )?;
        }

        for point in &self.points {
            writeln!(&mut file, "{} {} {}", point[0], point[1], point[2])?;
        }

        Ok(())
    }
}
