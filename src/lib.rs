extern crate byteorder;
extern crate cgmath;
extern crate disjoint_sets;
extern crate itertools;
extern crate nom;
extern crate rayon;

use cgmath::prelude::*;
use cgmath::{
    AbsDiffEq, Basis3, ElementWise, InnerSpace, Point2, Point3, Quaternion, Rotation, Rotation3,
    Vector3,
};
use nom::character::streaming::*;
use nom::error::VerboseError;
use nom::multi::count;
use nom::number::streaming::*;
use nom::sequence::*;
use nom::*;
use rayon::prelude::*;

use std::collections::HashMap;
use std::fs::File;
use std::io::prelude::*;
use std::io::{BufWriter, Write};
use std::iter::FromIterator;
use std::path::Path;
use std::str::FromStr;

use disjoint_sets::*;

use itertools::Itertools;

use byteorder::*;

#[derive(Debug)]
pub enum Error {
    ParseError(String),
    IOError(std::io::Error),
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::IOError(e)
    }
}

fn from_rodrigues(x: Vector3<f64>) -> Basis3<f64> {
    let theta2 = x.dot(x);
    if theta2 > cgmath::Rad::<f64>::default_epsilon() {
        let angle = cgmath::Rad(x.magnitude());
        let axis = x.normalize();
        cgmath::Basis3::from_axis_angle(axis, angle)
    } else {
        // taylor series approximation from ceres-solver
        Basis3::from(Quaternion::from(cgmath::Matrix3::new(
            1.0, x[2], -x[1], -x[2], 1.0, x[0], x[1], -x[0], 1.0,
        )))
    }
}

fn to_rodrigues(x: Basis3<f64>) -> Vector3<f64> {
    let q = Quaternion::from(x);
    let angle = 2.0 * q.s.acos();
    let axis = q.v / (1.0 - q.s * q.s).sqrt();
    axis.normalize() * angle
}

/// Camera expressed as Rx+t with intrinsics
/// The camera points down the negative z axis. Up is the positive y axis.
#[derive(Debug, Clone)]
pub struct Camera {
    loc: Vector3<f64>,    // t -- translation
    dir: Basis3<f64>,     // R -- rotation
    intrin: Vector3<f64>, // focal length, radial distortion x2
}

impl Camera {
    /// Project a point from the world into the camera coordinate system
    pub fn project_world(&self, p: &Point3<f64>) -> cgmath::Point3<f64> {
        self.dir.rotate_point(*p) + self.loc
    }

    /// Project a point from camera space into pixel coordinates
    pub fn project(&self, p: cgmath::Point3<f64>) -> cgmath::Point2<f64> {
        let p_ = cgmath::Vector2::new(-p.x / p.z, -p.y / p.z);
        let r = 1.0
            + self.distortion().0 * p_.magnitude2()
            + self.distortion().1 * p_.magnitude().powf(4.0);
        Point2::from_vec(self.focal_length() * r * p_)
    }

    pub fn from_vec(x: Vec<f64>) -> Self {
        Camera {
            dir: from_rodrigues(Vector3::new(x[0], x[1], x[2])),
            loc: Vector3::new(x[3], x[4], x[5]),
            intrin: Vector3::new(x[6], x[7], x[8]),
        }
    }

    pub fn to_vec(&self) -> Vec<f64> {
        let r = to_rodrigues(self.dir);
        vec![
            r.x,
            r.y,
            r.z,
            self.loc.x,
            self.loc.y,
            self.loc.z,
            self.intrin.x,
            self.intrin.y,
            self.intrin.z,
        ]
    }

    pub fn from_position_direction(
        position: Point3<f64>,
        dir: Basis3<f64>,
        intrin: Vector3<f64>,
    ) -> Self {
        Camera {
            loc: -1.0 * (dir.rotate_point(position)).to_vec(),
            dir: dir,
            intrin: intrin,
        }
    }

    pub fn center(&self) -> Point3<f64> {
        Point3::from_vec(-(self.dir.invert().rotate_vector(self.loc)))
    }

    pub fn rotation(&self) -> Basis3<f64> {
        self.dir
    }

    pub fn focal_length(&self) -> f64 {
        self.intrin[0]
    }

    pub fn distortion(&self) -> (f64, f64) {
        (self.intrin[1], self.intrin[2])
    }

    pub fn transform(
        &self,
        delta_dir: Basis3<f64>,
        delta_loc: Vector3<f64>,
        delta_intrin: Vector3<f64>,
    ) -> Camera {
        Camera {
            dir: self.dir * delta_dir,
            loc: -1.0
                * self
                    .rotation()
                    .rotate_point(self.center() + delta_loc)
                    .to_vec(),
            intrin: self.intrin + delta_intrin,
        }
    }

    pub fn to_world(&self, p: Point3<f64>) -> Point3<f64> {
        self.dir.invert().rotate_point(p - self.loc)
    }
}

#[test]
fn test_project_world() {
    let p = Point3::new(0.0, 0.0, -1.0);
    let c = Camera::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
    let p_camera = c.project_world(&p);
    assert!(p_camera.z < 0.0);
    assert!(p_camera.x == 0.0 && p_camera.y == 0.0);
}

#[test]
fn test_project() {
    let p = Point3::new(0.0, 0.0, -1.0);
    let c = Camera::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
    let uv = c.project(c.project_world(&p));
    assert!(uv.x == 0.0 && uv.y == 0.0);
}

#[test]
fn test_project_isomorphic() {
    let p = Point3::new(1.0, 3.0, -1.0);
    let c = Camera::from_vec(vec![3.0, 5.0, -2.0, 0.5, -0.2, 0.1, 1.0, 0.0, 0.0]);
    assert!(c.to_world(c.project_world(&p)).abs_diff_eq(&p, 1e-8));
}

pub fn total_reprojection_error(
    vis_graph: &Vec<Vec<(usize, (f64, f64))>>,
    cameras: &Vec<Camera>,
    points: &Vec<Point3<f64>>,
) -> f64 {
    cameras
        .par_iter()
        .zip(vis_graph)
        .map(|(camera, adj)| {
            adj.iter()
                .map(|(o, (u, v))| {
                    let p = camera.project(camera.project_world(&points[*o]));
                    (p.x - u).abs() + (p.y - v).abs()
                })
                .sum::<f64>()
        })
        .sum()
}

pub fn total_reprojection_error_l2(
    vis_graph: &Vec<Vec<(usize, (f64, f64))>>,
    cameras: &Vec<Camera>,
    points: &Vec<Point3<f64>>,
) -> f64 {
    cameras
        .par_iter()
        .zip(vis_graph)
        .map(|(camera, adj)| {
            adj.iter()
                .map(|(o, (u, v))| {
                    let p = camera.project(camera.project_world(&points[*o]));
                    (p.x - u).powf(2.0) + (p.y - v).powf(2.0)
                })
                .sum::<f64>()
        })
        .sum::<f64>()
        .sqrt()
        * 2.0
}

#[derive(Debug, Clone)]
pub struct BALProblem {
    pub cameras: Vec<Camera>,
    pub points: Vec<Point3<f64>>,
    pub vis_graph: Vec<Vec<(usize, (f64, f64))>>,
}

impl BALProblem {
    pub fn total_reprojection_error(&self) -> f64 {
        total_reprojection_error(&self.vis_graph, &self.cameras, &self.points)
    }

    pub fn total_reprojection_error_l2(&self) -> f64 {
        total_reprojection_error_l2(&self.vis_graph, &self.cameras, &self.points)
    }

    pub fn mean(&self) -> Vector3<f64> {
        let num = (self.cameras.len() + self.points.len()) as f64;
        self.cameras
            .iter()
            .map(|x| x.center().clone())
            .chain(self.points.clone().into_iter())
            .fold(Vector3::new(0.0, 0.0, 0.0), |a, b| a + b.to_vec() / num)
    }

    pub fn std(&self) -> Vector3<f64> {
        let num = (self.cameras.len() + self.points.len()) as f64;
        let mean = self.mean();
        (self
            .cameras
            .iter()
            .map(|x| x.center().clone())
            .chain(self.points.clone().into_iter())
            .map(|x| (x.to_vec() - mean).mul_element_wise(x.to_vec() - mean))
            .sum::<Vector3<f64>>()
            / num)
            .map(|x| x.sqrt())
    }

    pub fn new(
        cams: Vec<Camera>,
        points: Vec<Point3<f64>>,
        obs: Vec<(usize, usize, f64, f64)>,
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

    pub fn from_file_text(filepath: &Path) -> Result<BALProblem, Error> {
        fn parse_internal(input: &str) -> IResult<&str, BALProblem, VerboseError<&str>> {
            fn unsigned(input: &str) -> IResult<&str, usize, VerboseError<&str>> {
                nom::combinator::map_res(digit1, usize::from_str)(input)
            }

            let (input, num_cameras) = unsigned(input)?;
            let (input, _) = multispace0(input)?;
            let (input, num_points) = unsigned(input)?;
            let (input, _) = multispace0(input)?;
            let (input, num_observations) = unsigned(input)?;
            let (input, _) = multispace0(input)?;

            let (input, observations) = count(
                tuple((
                    preceded(multispace0, unsigned),
                    preceded(multispace0, unsigned),
                    preceded(multispace0, double),
                    preceded(multispace0, double),
                )),
                num_observations,
            )(input)?;

            let camera = nom::combinator::map(count(preceded(multispace0, double), 9), |x| {
                Camera::from_vec(x)
            });
            let (input, cameras) = count(camera, num_cameras)(input)?;
            let point = nom::combinator::map(count(preceded(multispace0, double), 3), |x| {
                Point3::new(x[0], x[1], x[2])
            });
            let (input, points) = count(point, num_points)(input)?;

            Ok((input, BALProblem::new(cameras, points, observations)))
        }

        let mut file = File::open(filepath)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;
        parse_internal(contents.as_ref())
            .map(|x| x.1)
            .map_err(|x| match x {
                nom::Err::Error(e) | nom::Err::Failure(e) => {
                    Error::ParseError(nom::error::convert_error(contents.as_ref(), e))
                }
                nom::Err::Incomplete(x) => Error::ParseError(format!("{:?}", x)),
            })
    }

    pub fn from_file_binary(filepath: &Path) -> Result<BALProblem, Error> {
        fn parse_internal(input: &[u8]) -> IResult<&[u8], BALProblem, VerboseError<&[u8]>> {
            let (input, num_cameras) = be_u64(input)?;
            let (input, num_points) = be_u64(input)?;
            let (input, _num_observations) = be_u64(input)?;

            let (input, observations) = count(
                |input| {
                    let (input, num_obs) = be_u64(input)?;
                    let (input, obs) = count(
                        tuple((
                            nom::combinator::map(be_u64, |x| x as usize),
                            tuple((be_f64, be_f64)),
                        )),
                        num_obs as usize,
                    )(input)?;
                    Ok((input, obs))
                },
                num_cameras as usize,
            )(input)?;

            let (input, cameras) = count(
                |input| {
                    let (input, v) = count(be_f64, 9)(input)?;
                    Ok((input, Camera::from_vec(v)))
                },
                num_cameras as usize,
            )(input)?;

            let (input, points) = count(
                |input| {
                    let (input, p) = count(be_f64, 3)(input)?;
                    Ok((input, Point3::new(p[0], p[1], p[2])))
                },
                num_points as usize,
            )(input)?;

            Ok((
                input,
                BALProblem {
                    cameras: cameras,
                    points: points,
                    vis_graph: observations,
                },
            ))
        }

        let mut file = File::open(filepath)?;
        let mut contents = Vec::new();
        file.read_to_end(&mut contents)?;

        parse_internal(contents.as_slice())
            .map(|x| x.1)
            .map_err(|x| match x {
                nom::Err::Error(_) | nom::Err::Failure(_) => {
                    Error::ParseError("Binary parse error".to_string())
                }
                nom::Err::Incomplete(x) => Error::ParseError(format!("{:?}", x)),
            })
    }

    pub fn from_file(path: &Path) -> Result<BALProblem, Error> {
        match path.extension().unwrap().to_str().unwrap() {
            "bal" => Self::from_file_text(path),
            "bbal" => Self::from_file_binary(path),
            ext => Err(Error::IOError(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("unknown file extension {}", ext),
            ))),
        }
    }

    pub fn num_points(&self) -> usize {
        self.points.len()
    }

    pub fn num_cameras(&self) -> usize {
        self.cameras.len()
    }

    pub fn num_observations(&self) -> usize {
        self.vis_graph.iter().map(|x| x.len()).sum()
    }

    pub fn verify(&self) {
        if self.vis_graph.len() > self.num_cameras() {
            println!(
                "Have more observations than camears. {} vs {}.",
                self.vis_graph.len(),
                self.num_cameras()
            );
        }
        for (i, obs) in self.vis_graph.iter().enumerate() {
            for (j, _) in obs.iter() {
                if j >= &self.num_points() {
                    println!(
                        "Invalid observation of point {} ({}) from camera {}",
                        j,
                        self.num_points(),
                        i
                    );
                }
            }
        }
    }

    pub fn subset(&self, ci: &[usize], pi: &[usize]) -> Self {
        let cameras = ci
            .iter()
            .map(|i| self.cameras[*i].clone())
            .collect::<Vec<_>>();
        let points = pi
            .iter()
            .map(|i| self.points[*i].clone())
            .collect::<Vec<_>>();

        // use i64 here so we can mark points that aren't in the final set
        let mut point_indices: Vec<i64> = vec![-1; self.points.len()];
        for (i, p) in pi.iter().enumerate() {
            point_indices[*p] = i as i64;
        }

        let obs = ci
            .iter()
            .map(|i| self.vis_graph[*i].clone())
            .map(|obs| {
                obs.iter()
                    .filter(|(i, _)| point_indices[*i] >= 0)
                    .map(|(i, uv)| (point_indices[*i] as usize, uv.clone()))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        BALProblem {
            cameras: cameras,
            points: points,
            vis_graph: obs,
        }
    }

    pub fn remove_singletons(&self) -> Self {
        // remove cameras that see less than 4 points
        let ci = self
            .vis_graph
            .iter()
            .enumerate()
            .filter(|(_, v)| v.len() > 0)
            .map(|(i, _)| i)
            .collect::<Vec<_>>();

        let mut point_count: Vec<i64> = vec![0; self.points.len()];
        // TODO: skip cameras that we have already removed
        for obs in self.vis_graph.iter() {
            for (i, _) in obs.iter() {
                point_count[*i] += 1;
            }
        }

        // remove points seen less than twice
        let pi = point_count
            .iter()
            .enumerate()
            .filter(|(_, c)| **c > 0)
            .map(|(i, _)| i)
            .collect::<Vec<_>>();

        self.subset(ci.as_slice(), pi.as_slice())
    }

    /// Get the largest connected component of cameras and points.
    pub fn largest_connected_component(&self) -> Self {
        if self.num_cameras() <= 0 {
            return (*self).clone();
        }
        let mut uf = UnionFind::new(self.num_points() + self.num_cameras());

        // point index is num_cameras + point id
        for (i, obs) in self.vis_graph.iter().enumerate() {
            for (j, _) in obs {
                let p = j + self.num_cameras();
                if !uf.equiv(i, p) {
                    uf.union(i, p);
                }
            }
        }

        let sets = uf.to_vec();

        // find largest set
        let mut hm = HashMap::new();
        for i in 0..self.num_cameras() {
            let x = hm.entry(sets[i]).or_insert(0);
            *x += 1;
        }
        let lcc_id = *(hm
            .iter()
            .sorted_by(|a, b| Ord::cmp(&b.1, &a.1))
            .next()
            .unwrap()
            .0);

        // compute component
        // new cameras and points
        let cameras = self
            .cameras
            .iter()
            .zip(sets.iter())
            .filter(|x| *x.1 == lcc_id)
            .map(|x| x.0.clone())
            .collect::<Vec<_>>();
        let points = self
            .points
            .iter()
            .zip(sets[self.num_cameras()..].iter())
            .filter(|x| *x.1 == lcc_id)
            .map(|x| x.0.clone())
            .collect::<Vec<_>>();

        // map from old id to new
        let point_ids = sets[self.num_cameras()..(self.num_cameras() + self.num_points())]
            .iter()
            .enumerate()
            .filter(|x| *x.1 == lcc_id)
            .map(|x| x.0);
        let point_map =
            HashMap::<usize, usize>::from_iter(point_ids.enumerate().map(|(x, y)| (y, x)));
        // new camera id is implicitly handled by filtering
        let vis_graph = self
            .vis_graph
            .iter()
            .enumerate()
            .filter(|x| sets[x.0] == lcc_id)
            .map(|(_, obs)| {
                obs.into_iter()
                    .filter(|x| sets[x.0] == lcc_id)
                    .map(|(i, p)| (point_map[&i], p.clone()))
                    .collect()
            })
            .collect();

        BALProblem {
            cameras: cameras,
            points: points,
            vis_graph: vis_graph,
        }
    }

    /// Construct the largest connected component that contains cameras viewing 4 or more points
    /// and points viewed at least twice.
    pub fn cull(&self) -> Self {
        let mut nc = self.num_cameras();
        let mut np = self.num_points();
        let mut culled = self.largest_connected_component().remove_singletons();
        while culled.num_cameras() != nc || culled.num_points() != np {
            nc = culled.num_cameras();
            np = culled.num_points();
            culled = culled.largest_connected_component().remove_singletons();
        }

        for (i, obs) in culled.vis_graph.iter().enumerate() {
            if obs.len() < 2 {
                println!("{} {}", i, obs.len());
            }
        }

        culled
    }

    pub fn write_text(&self, path: &std::path::Path) -> Result<(), std::io::Error> {
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
            writeln!(&mut file, "{}", camera.to_vec().iter().join(" "))?;
        }

        for point in &self.points {
            writeln!(&mut file, "{} {} {}", point[0], point[1], point[2])?;
        }

        Ok(())
    }

    pub fn write_binary(&self, path: &std::path::Path) -> Result<(), std::io::Error> {
        let mut file = BufWriter::new(File::create(path).unwrap());
        file.write_u64::<BigEndian>(self.cameras.len() as u64)?;
        file.write_u64::<BigEndian>(self.points.len() as u64)?;
        file.write_u64::<BigEndian>(self.vis_graph.iter().map(|x| x.len()).sum::<usize>() as u64)?;

        for obs in self.vis_graph.iter() {
            file.write_u64::<BigEndian>(obs.len() as u64)?;
            for (p, (u, v)) in obs {
                file.write_u64::<BigEndian>(*p as u64)?;
                file.write_f64::<BigEndian>(*u as f64)?;
                file.write_f64::<BigEndian>(*v as f64)?;
            }
        }

        for camera in &self.cameras {
            for x in camera.to_vec().into_iter() {
                file.write_f64::<BigEndian>(x)?;
            }
        }

        for point in &self.points {
            file.write_f64::<BigEndian>(point[0] as f64)?;
            file.write_f64::<BigEndian>(point[1] as f64)?;
            file.write_f64::<BigEndian>(point[2] as f64)?;
        }

        Ok(())
    }

    pub fn write(&self, path: &std::path::Path) -> Result<(), std::io::Error> {
        match path.extension().unwrap().to_str().unwrap() {
            "bal" => self.write_text(path),
            "bbal" => self.write_binary(path),
            ext => Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("unknown file extension {}", ext),
            )),
        }
    }
}
