extern crate byteorder;
extern crate cgmath;
extern crate disjoint_sets;
extern crate itertools;
extern crate nom;
extern crate rayon;

use cgmath::{ElementWise, InnerSpace, Vector3};
use nom::character::streaming::*;
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
    ParseError,
    IOError(std::io::Error),
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
    pub loc: Vector3<f64>,
    pub dir: Vector3<f64>,    // Rodriguez
    pub intrin: Vector3<f64>, // Rodriguez
    pub img_size: (usize, usize),
}

impl Camera {
    /// Project a point from the world into the camera coordinate system
    pub fn project_world(&self, p: &Vector3<f64>) -> cgmath::Vector3<f64> {
        let angle = cgmath::Rad(self.dir.magnitude());
        let axis = self.dir.normalize();
        cgmath::Matrix3::from_axis_angle(axis, angle) * p + self.loc
    }

    /// Project a point from camera space into pixel coordinates
    pub fn project(&self, p: cgmath::Vector3<f64>) -> cgmath::Vector2<f64> {
        let p_ = cgmath::Vector2::new(-p.x / p.z, -p.y / p.z);
        let r = 1.0
            + self.distortion().0 * p_.magnitude2()
            + self.distortion().1 * p_.magnitude().powf(4.0);
        self.focal_length() * r * p_
    }

    pub fn from_vec(x: Vec<f64>) -> Self {
        Camera {
            dir: Vector3::new(x[0], x[1], x[2]),
            loc: Vector3::new(x[3], x[4], x[5]),
            intrin: Vector3::new(x[6], x[7], x[8]),
            img_size: (1024, 1024),
        }
    }

    pub fn focal_length(&self) -> f64 {
        self.intrin[0]
    }

    pub fn distortion(&self) -> (f64, f64) {
        (self.intrin[1], self.intrin[2])
    }
}

pub fn total_reprojection_error(
    vis_graph: &Vec<Vec<(usize, (f64, f64))>>,
    cameras: &Vec<Camera>,
    points: &Vec<Vector3<f64>>,
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
    points: &Vec<Vector3<f64>>,
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

#[derive(Debug)]
pub struct BALProblem {
    pub cameras: Vec<Camera>,
    pub points: Vec<Vector3<f64>>,
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
            .map(|x| &x.loc)
            .chain(self.points.iter())
            .fold(Vector3::new(0.0, 0.0, 0.0), |a, &b| a + b / num)
    }

    pub fn std(&self) -> Vector3<f64> {
        let num = (self.cameras.len() + self.points.len()) as f64;
        let mean = self.mean();
        (self
            .cameras
            .iter()
            .map(|x| &x.loc)
            .chain(self.points.iter())
            .map(|x| (x - mean).mul_element_wise(x - mean))
            .sum::<Vector3<f64>>()
            / num)
            .map(|x| x.sqrt())
    }

    pub fn new(
        cams: Vec<Camera>,
        points: Vec<Vector3<f64>>,
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

    // pub fn from_file(filepath: &Path) -> Result<BALProblem, Error> {
    //     named!(integer<&[u8], usize>,
    //            map_res!(map_res!(digit1, std::str::from_utf8), usize::from_str));
    //
    //     named!(header<&[u8], (usize, usize, usize)>, ws!(tuple!(integer, integer, integer)));
    //
    //     named!(observation<&[u8], (usize, usize, f64, f64)>,
    //     do_parse!(c_i: integer >>
    //               space1 >>
    //               p_i : integer >>
    //               space1 >>
    //               obs_x: float >>
    //               space1 >>
    //               obs_y: float >>
    //               ((c_i, p_i, obs_x, obs_y))
    //               ));
    //
    //     fn from_vec(x: Vec<f64>) -> Result<Camera, u8> {
    //         Ok(Camera::from_vec(x))
    //     }
    //
    //     named!(camera<&[u8], Camera>,
    //            map_res!(count!(preceded!(space0, float), 9), from_vec));
    //
    //     named!(point<&[u8], Vector3<f64> >,
    //     do_parse!(x: float >>
    //               space1 >>
    //               y: float >>
    //               space1 >>
    //               z: float >>
    //               (Vector3::new(x, y, z))
    //               ));
    //
    //     named!(bal_problem<&[u8], BALProblem>,
    //     do_parse!(hdr: header >>
    //               obs: count!(preceded!(opt!(tag!("\n")), observation), hdr.2) >>
    //               cams: count!(dbg_dmp!(preceded!(tag!("\n"), camera)), hdr.0) >>
    //               pts: dbg_dmp!(count!(preceded!(tag!("\n"), point), hdr.1)) >>
    //               (BALProblem::new(cams, pts, obs))
    //           ));
    //
    //     let mut file = File::open(filepath)?;
    //     let mut contents = Vec::new();
    //     file.read_to_end(&mut contents)?;
    //     bal_problem(contents.as_slice())
    //         .map(|x| x.1)
    //         .map_err(Error::from)
    // }

    pub fn from_file_text(filepath: &Path) -> Result<BALProblem, Error> {
        fn parse_internal(input: &str) -> IResult<&str, BALProblem> {
            fn float(input: &str) -> IResult<&str, f64> {
                nom::combinator::map_res(digit1, f64::from_str)(input)
            }
            fn unsigned(input: &str) -> IResult<&str, usize> {
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
                    preceded(multispace0, float),
                    preceded(multispace0, float),
                )),
                num_observations,
            )(input)?;

            let camera = nom::combinator::map(count(preceded(multispace0, float), 9), |x| {
                Camera::from_vec(x)
            });
            let (input, cameras) = count(camera, num_cameras)(input)?;
            let point = nom::combinator::map(count(preceded(multispace0, float), 3), |x| {
                Vector3::new(x[0], x[1], x[2])
            });
            let (input, points) = count(point, num_points)(input)?;

            Ok((input, BALProblem::new(cameras, points, observations)))
        }

        let mut file = File::open(filepath)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;
        parse_internal(contents.as_ref())
            .map(|x| x.1)
            .map_err(Error::from)
    }

    pub fn from_file_binary(filepath: &Path) -> Result<BALProblem, Error> {
        fn parse_internal(input: &[u8]) -> IResult<&[u8], BALProblem> {
            let (input, num_cameras) = be_u64(input)?;
            let (input, num_points) = be_u64(input)?;
            let (input, num_observations) = be_u64(input)?;

            let (input, observations) = count(
                |input| {
                    let (input, c_i) = be_u64(input)?;
                    let (input, p_i) = be_u64(input)?;
                    let (input, obs_x) = be_f64(input)?;
                    let (input, obs_y) = be_f64(input)?;
                    Ok((input, (c_i as usize, p_i as usize, obs_x, obs_y)))
                },
                num_observations as usize,
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
                    Ok((input, Vector3::new(p[0], p[1], p[2])))
                },
                num_points as usize,
            )(input)?;

            Ok((input, BALProblem::new(cameras, points, observations)))
        }

        let mut file = File::open(filepath)?;
        let mut contents = Vec::new();
        file.read_to_end(&mut contents)?;

        parse_internal(contents.as_slice())
            .map(|x| x.1)
            .map_err(Error::from)
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

    /// Get the largest connected component of cameras and points.
    pub fn largest_connected_component(&self) -> Self {
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

    pub fn write_binary(&self, path: &std::path::Path) -> Result<(), std::io::Error> {
        let mut file = BufWriter::new(File::create(path).unwrap());
        file.write_u64::<BigEndian>(self.cameras.len() as u64)?;
        file.write_u64::<BigEndian>(self.points.len() as u64)?;
        file.write_u64::<BigEndian>(self.vis_graph.iter().map(|x| x.len() as u64).sum::<u64>())?;

        for (i, obs) in self.vis_graph.iter().enumerate() {
            for (p, (u, v)) in obs {
                file.write_u64::<BigEndian>(i as u64)?;
                file.write_u64::<BigEndian>(*p as u64)?;
                file.write_f64::<BigEndian>(*u as f64)?;
                file.write_f64::<BigEndian>(*v as f64)?;
            }
        }

        for camera in &self.cameras {
            file.write_f64::<BigEndian>(camera.dir[0] as f64)?;
            file.write_f64::<BigEndian>(camera.dir[1] as f64)?;
            file.write_f64::<BigEndian>(camera.dir[2] as f64)?;
            file.write_f64::<BigEndian>(camera.loc[0] as f64)?;
            file.write_f64::<BigEndian>(camera.loc[1] as f64)?;
            file.write_f64::<BigEndian>(camera.loc[2] as f64)?;
            file.write_f64::<BigEndian>(camera.intrin[0] as f64)?;
            file.write_f64::<BigEndian>(camera.intrin[1] as f64)?;
            file.write_f64::<BigEndian>(camera.intrin[2] as f64)?;
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
