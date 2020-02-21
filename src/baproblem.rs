extern crate byteorder;
extern crate cgmath;
extern crate disjoint_sets;
extern crate itertools;
extern crate nom;

use byteorder::*;
use cgmath::prelude::*;
use cgmath::{
    AbsDiffEq, Basis3, ElementWise, InnerSpace, Point2, Point3, Quaternion, Rotation, Rotation3,
    Vector3,
};
use disjoint_sets::*;
use itertools::Itertools;
use nom::character::streaming::*;
use nom::error::VerboseError;
use nom::multi::count;
use nom::number::streaming::*;
use nom::sequence::*;
use nom::*;

use std::collections::HashMap;
use std::fs::File;
use std::io::prelude::*;
use std::io::{BufWriter, Write};
use std::iter::FromIterator;
use std::path::Path;
use std::str::FromStr;

#[derive(Debug)]
pub enum Error {
    ParseError(String),
    EmptyProblem(String),
    IOError(std::io::Error),
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::IOError(e)
    }
}

/// Convert Rodrigues vector to a rotation.
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

/// Convert rotation to Rodrigues vector.
fn to_rodrigues(x: Basis3<f64>) -> Vector3<f64> {
    let q = Quaternion::from(x);
    let angle = 2.0 * q.s.acos();
    let axis = q.v / (1.0 - q.s * q.s).sqrt();
    axis.normalize() * angle
}

/// A projective camera.
///
/// The camera must have a center and a way to project points to and from the camera frame.
pub trait Camera {
    /// Project a point from the world into the camera coordinate system
    fn project_world(&self, p: &Point3<f64>) -> Point3<f64>;

    /// Project a point from camera space into pixel coordinates
    fn project(&self, p: cgmath::Point3<f64>) -> cgmath::Point2<f64>;

    /// Create a camera from a position and direction.
    fn from_position_direction(position: Point3<f64>, dir: Basis3<f64>) -> Self;

    /// Center of the camera.
    fn center(&self) -> Point3<f64>;

    /// Transform a camera with a rotational and translational modification.
    fn transform(self, delta_dir: Basis3<f64>, delta_loc: Vector3<f64>) -> Self;

    /// Project a point into this cameras frame of reference.
    fn to_world(&self, p: Point3<f64>) -> Point3<f64>;
}

/// Camera expressed as Rx+t with intrinsics.
///
/// The camera points down the negative z axis. Up is the positive y axis.
#[derive(Debug, Clone)]
pub struct SnavelyCamera {
    /// Translational parameter `t`
    pub loc: Vector3<f64>,
    /// Rotational parameter `R`
    pub dir: Basis3<f64>,
    /// Intrinsics `intrin[0]` is the focal length, `intrin[1]` is the squared distortion, and `intrin[2]` is the quadratic distortion.
    pub intrin: Vector3<f64>,
}

impl Camera for SnavelyCamera {
    fn project_world(&self, p: &Point3<f64>) -> cgmath::Point3<f64> {
        self.dir.rotate_point(*p) + self.loc
    }

    fn project(&self, p: cgmath::Point3<f64>) -> cgmath::Point2<f64> {
        let p_ = cgmath::Vector2::new(-p.x / p.z, -p.y / p.z);
        let r = 1.0
            + self.distortion().0 * p_.magnitude2()
            + self.distortion().1 * p_.magnitude().powf(4.0);
        Point2::from_vec(self.focal_length() * r * p_)
    }

    fn from_position_direction(position: Point3<f64>, dir: Basis3<f64>) -> Self {
        SnavelyCamera {
            loc: -1.0 * (dir.rotate_point(position)).to_vec(),
            dir: dir,
            intrin: Vector3::new(1., 0., 0.),
        }
    }

    fn center(&self) -> Point3<f64> {
        Point3::from_vec(-(self.dir.invert().rotate_vector(self.loc)))
    }

    fn transform(self, delta_dir: Basis3<f64>, delta_loc: Vector3<f64>) -> Self {
        SnavelyCamera {
            dir: self.dir * delta_dir,
            loc: -1.0 * self.dir.rotate_point(self.center() + delta_loc).to_vec(),
            intrin: self.intrin,
        }
    }

    fn to_world(&self, p: Point3<f64>) -> Point3<f64> {
        self.dir.invert().rotate_point(p - self.loc)
    }
}

impl SnavelyCamera {
    /// Parse a camera from a vector of parameters. Order is rotation as a 3 element Rodrigues vector, translation, intrinsics.
    pub fn from_vec(x: Vec<f64>) -> Self {
        SnavelyCamera {
            dir: from_rodrigues(Vector3::new(x[0], x[1], x[2])),
            loc: Vector3::new(x[3], x[4], x[5]),
            intrin: Vector3::new(x[6], x[7], x[8]),
        }
    }

    /// Parse a camera to a vector of parameters. Order is rotation as a 3 element Rodrigues vector, translation, intrinsics.
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

    /// R parameter of camera.
    pub fn rotation(&self) -> Basis3<f64> {
        self.dir
    }

    pub fn focal_length(&self) -> f64 {
        self.intrin[0]
    }

    pub fn distortion(&self) -> (f64, f64) {
        (self.intrin[1], self.intrin[2])
    }

    /// Adjust the intrinsics of the camera as `intrin + delta`.
    pub fn modify_intrin(self, delta: Vector3<f64>) -> Self {
        SnavelyCamera {
            dir: self.dir,
            loc: self.loc,
            intrin: self.intrin + delta,
        }
    }
}

#[test]
fn test_project_world() {
    let p = Point3::new(0.0, 0.0, -1.0);
    let c = SnavelyCamera::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
    let p_camera = c.project_world(&p);
    assert!(p_camera.z < 0.0);
    assert!(p_camera.x == 0.0 && p_camera.y == 0.0);
}

#[test]
fn test_project() {
    let p = Point3::new(0.0, 0.0, -1.0);
    let c = SnavelyCamera::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
    let uv = c.project(c.project_world(&p));
    assert!(uv.x == 0.0 && uv.y == 0.0);
}

#[test]
fn test_project_isomorphic() {
    let p = Point3::new(1.0, 3.0, -1.0);
    let c = SnavelyCamera::from_vec(vec![3.0, 5.0, -2.0, 0.5, -0.2, 0.1, 1.0, 0.0, 0.0]);
    assert!(c.to_world(c.project_world(&p)).abs_diff_eq(&p, 1e-8));
}

/// Bundle adjustment problem composed of cameras, points, and observations of points by cameras.
///
/// Observations are stored as an array of arrays where `v[i][j] = (k, (u, v))` indicates that camera
/// `i` sees point `k` at `(u, v)` in the camera frame.
#[derive(Debug, Clone)]
pub struct BAProblem<C: Camera> {
    pub cameras: Vec<C>,
    pub points: Vec<Point3<f64>>,
    pub vis_graph: Vec<Vec<(usize, (f64, f64))>>,
}

impl<C: Camera> BAProblem<C> {
    /// Amount of reprojection error in the problem. Computed as the `norm`-norm of the difference
    /// of all observed points from their projection.
    pub fn total_reprojection_error(&self, norm: f64) -> f64 {
        self.cameras
            .iter()
            .zip(&self.vis_graph)
            .map(|(camera, adj)| {
                adj.iter()
                    .map(|(o, (u, v))| {
                        let p = camera.project(camera.project_world(&self.points[*o]));
                        (p.x - u).abs().powf(norm) + (p.y - v).abs().powf(norm)
                    })
                    .sum::<f64>()
            })
            .sum::<f64>()
            .powf(1. / norm)
    }

    /// Center of mass of cameras and points.
    pub fn mean(&self) -> Vector3<f64> {
        let num = (self.cameras.len() + self.points.len()) as f64;
        self.cameras
            .iter()
            .map(|x| x.center().clone())
            .chain(self.points.clone().into_iter())
            .fold(Vector3::new(0.0, 0.0, 0.0), |a, b| a + b.to_vec() / num)
    }

    /// Standard deviation of cameras and points from the center of mass.
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

    /// Smallest and largest coordinates of the problem.
    pub fn extent(&self) -> (Vector3<f64>, Vector3<f64>) {
        let min = self
            .cameras
            .iter()
            .map(|x| x.center().clone())
            .chain(self.points.clone().into_iter())
            .fold(
                Vector3::new(std::f64::INFINITY, std::f64::INFINITY, std::f64::INFINITY),
                |x, y| Vector3::new(x.x.min(y.x), x.y.min(y.y), x.z.min(y.z)),
            );
        let max = self
            .cameras
            .iter()
            .map(|x| x.center().clone())
            .chain(self.points.clone().into_iter())
            .fold(
                Vector3::new(
                    -std::f64::INFINITY,
                    -std::f64::INFINITY,
                    -std::f64::INFINITY,
                ),
                |x, y| Vector3::new(x.x.max(y.x), x.y.max(y.y), x.z.max(y.z)),
            );
        (min, max)
    }

    /// Dimensions in x,y,z of the problem.
    pub fn dimensions(&self) -> Vector3<f64> {
        let (min, max) = self.extent();
        max - min
    }

    /// Create a new bundle adjustment problem from a set a cameras, points, and observations.
    /// Observations are a tuple of camera index, point index, u, v where the camera sees the point
    /// at u,v.
    pub fn new(cams: Vec<C>, points: Vec<Point3<f64>>, obs: Vec<(usize, usize, f64, f64)>) -> Self {
        let mut vis_graph = vec![Vec::new(); cams.len()];
        for (cam_i, p_i, obs_x, obs_y) in obs {
            assert!(cam_i < cams.len());
            assert!(p_i < points.len());
            vis_graph[cam_i].push((p_i, (obs_x, obs_y)));
        }

        BAProblem {
            cameras: cams,
            points: points,
            vis_graph: vis_graph,
        }
    }

    /// Create a new bundle adjustment problem from a set a cameras, points, and observations.
    /// Observations are a vector containing vectors of the points seen by the camera at the
    /// respective index.
    pub fn from_visibility(
        cams: Vec<C>,
        points: Vec<Point3<f64>>,
        obs: Vec<Vec<(usize, (f64, f64))>>,
    ) -> Self {
        assert!(cams.len() == obs.len());
        for o in &obs {
            for (ci, _) in o {
                assert!(ci < &points.len());
            }
        }
        BAProblem {
            cameras: cams,
            points: points,
            vis_graph: obs,
        }
    }

    pub fn num_points(&self) -> usize {
        self.points.len()
    }

    pub fn num_cameras(&self) -> usize {
        self.cameras.len()
    }

    /// Number of camera-point observations.
    pub fn num_observations(&self) -> usize {
        self.vis_graph.iter().map(|x| x.len()).sum()
    }
}

impl<C: Camera + Clone> BAProblem<C> {
    /// Select a subset of the problem with camera indices in `ci` and point indices in `pi`.
    pub fn subset(self, ci: &[usize], pi: &[usize]) -> Self {
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

        BAProblem {
            cameras: cameras,
            points: points,
            vis_graph: obs,
        }
    }

    /// Remove cameras that see less than 4 points and points seen less than twice.
    pub fn remove_singletons(self) -> Self {
        // remove cameras that see less than 4 points
        let ci = self
            .vis_graph
            .iter()
            .enumerate()
            .filter(|(_, v)| v.len() > 3)
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
            .filter(|(_, c)| **c > 1)
            .map(|(i, _)| i)
            .collect::<Vec<_>>();

        self.subset(ci.as_slice(), pi.as_slice())
    }

    /// Get the largest connected component of cameras and points.
    pub fn largest_connected_component(self) -> Self {
        if self.num_cameras() <= 0 {
            return self;
        }

        let num_cameras = self.num_cameras();
        let num_points = self.num_points();

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

        // find largest set
        let sets = uf.to_vec();
        let mut hm = HashMap::new();
        for i in 0..num_cameras {
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
            .into_iter()
            .zip(sets.iter())
            .filter(|x| *x.1 == lcc_id)
            .map(|x| x.0)
            .collect::<Vec<_>>();
        let points = self
            .points
            .into_iter()
            .zip(sets[num_cameras..].iter())
            .filter(|x| *x.1 == lcc_id)
            .map(|x| x.0)
            .collect::<Vec<_>>();

        // map from old id to new
        let point_ids = sets[num_cameras..(num_cameras + num_points)]
            .iter()
            .enumerate()
            .filter(|x| *x.1 == lcc_id)
            .map(|x| x.0);
        let point_map =
            HashMap::<usize, usize>::from_iter(point_ids.enumerate().map(|(x, y)| (y, x)));
        // new camera id is implicitly handled by filtering
        let vis_graph = self
            .vis_graph
            .into_iter()
            .enumerate()
            .filter(|x| sets[x.0] == lcc_id)
            .map(|(_, obs)| {
                obs.into_iter()
                    .filter(|x| sets[x.0] == lcc_id)
                    .map(|(i, p)| (point_map[&i], p))
                    .collect()
            })
            .collect();

        BAProblem {
            cameras: cameras,
            points: points,
            vis_graph: vis_graph,
        }
    }

    /// Construct the largest connected component that contains cameras viewing 4 or more points
    /// and points viewed at least twice.
    pub fn cull(self) -> Self {
        let mut nc = self.num_cameras();
        let mut np = self.num_points();
        let mut culled = self.largest_connected_component().remove_singletons();
        while culled.num_cameras() != nc || culled.num_points() != np {
            nc = culled.num_cameras();
            np = culled.num_points();
            culled = culled.largest_connected_component().remove_singletons();
        }

        culled
    }
}

impl BAProblem<SnavelyCamera> {
    /// Parse a bundle adjustment problem from a file in the Bundle Adjustment in the Large text
    /// file format.
    ///
    /// ```txt
    /// <num_cameras> <num_points> <num_observations>
    /// <camera_index_1> <point_index_1> <x_1> <y_1>
    /// ...
    /// <camera_index_num_observations> <point_index_num_observations> <x_num_observations> <y_num_observations>
    /// <camera_1>
    /// ...
    /// <camera_num_cameras>
    /// <point_1>
    /// ...
    /// <point_num_points>
    /// ```
    /// where cameras are:
    /// ```txt
    /// <R_1>
    /// <R_2>
    /// <R_3>
    /// <t_1>
    /// <t_2>
    /// <t_3>
    /// <focal length>
    /// <distortion^2>
    /// <distortion^4>
    /// ```
    pub fn from_file_text(filepath: &Path) -> Result<Self, Error> {
        fn parse_internal(
            input: &str,
        ) -> IResult<&str, BAProblem<SnavelyCamera>, VerboseError<&str>> {
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
                SnavelyCamera::from_vec(x)
            });
            let (input, cameras) = count(camera, num_cameras)(input)?;
            let point = nom::combinator::map(count(preceded(multispace0, double), 3), |x| {
                Point3::new(x[0], x[1], x[2])
            });
            let (input, points) = count(point, num_points)(input)?;

            Ok((input, BAProblem::new(cameras, points, observations)))
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

    /// Parse a bundle adjustment problem from a file in the Bundle Adjustment in the Large binary
    /// file format.
    pub fn from_file_binary(filepath: &Path) -> Result<Self, Error> {
        fn parse_internal(
            input: &[u8],
        ) -> IResult<&[u8], BAProblem<SnavelyCamera>, VerboseError<&[u8]>> {
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
                    Ok((input, SnavelyCamera::from_vec(v)))
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
                BAProblem {
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

    /// Parse a bundle adjustment problem from a file in the Bundle Adjustment in the Large format.
    /// Supports both binary and text formats.
    pub fn from_file(path: &Path) -> Result<Self, Error> {
        match path.extension().unwrap().to_str().unwrap() {
            "bal" => Self::from_file_text(path),
            "bbal" => Self::from_file_binary(path),
            ext => Err(Error::IOError(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("unknown file extension {}", ext),
            ))),
        }
    }

    /// Write problem in Bundle Adjustment in the Large text format.
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

    /// Write problem in Bundle Adjustment in the Large binary format.
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

    /// Write BAProblem to a file in BAL format. Text or binary format is automatically chosen from
    /// the filename extension. `.bal` -> text, `.bbal` -> binary.
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

impl<C> std::fmt::Display for BAProblem<C>
where
    C: Camera,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Bundle Adjustment Problem with {} cameras, {} points, and {} observations",
            self.num_cameras(),
            self.num_points(),
            self.num_observations()
        )
    }
}
