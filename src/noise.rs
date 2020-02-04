extern crate rand;
use rand::distributions::{Distribution, Normal, WeightedIndex};
use rand::prelude::*;
use rand::seq::SliceRandom;
use rand::{thread_rng, Rng};

use crate::baproblem::*;

extern crate cgmath;
use cgmath::*;

extern crate itertools;
use itertools::Itertools;

extern crate rstar;
use rstar::RTree;

use std::collections::HashMap;
use std::iter::FromIterator;
fn unit_random() -> Vector3<f64> {
    let r = Normal::new(0.0, 1.0);
    Vector3::new(
        r.sample(&mut rand::thread_rng()) as f64,
        r.sample(&mut rand::thread_rng()) as f64,
        r.sample(&mut rand::thread_rng()) as f64,
    )
    .normalize()
}

// TODO: add twisting to the noise
pub fn add_drift(bal: BALProblem, strength: f64, angle_strength: f64, std: f64) -> BALProblem {
    // Choose the drift direction to be in line with the largest standard deviation.
    let dir = bal.std().normalize();

    let origin = bal
        .cameras
        .iter()
        .map(|c| c.center())
        .chain(bal.points.clone().into_iter())
        .fold1(|x, y| {
            if x.distance(EuclideanSpace::origin()) < y.distance(EuclideanSpace::origin()) {
                x
            } else {
                y
            }
        })
        .unwrap();

    let r = Normal::new(1.0, std.into());
    let bal_std = bal.std().magnitude();

    let drift_noise = |x: Point3<f64>| {
        let distance = (x - origin).magnitude();
        let v = r.sample(&mut rand::thread_rng()) as f64;
        dir * strength * v * bal_std * distance * distance
    };
    let drift_angle = |x: Point3<f64>| {
        let distance = (x - origin).magnitude();
        let v = r.sample(&mut rand::thread_rng()) as f64;
        angle_strength * v * distance.powf(1.2)
    };
    let cameras = bal
        .cameras
        .iter()
        .map(|c| {
            c.transform(
                Basis3::from_angle_x(Rad(drift_angle(c.center()))),
                drift_noise(c.center()),
                Vector3::new(0.0, 0.0, 0.0),
            )
        })
        .collect();

    let points = bal.points.iter().map(|p| p + drift_noise(*p)).collect();

    BALProblem {
        cameras: cameras,
        points: points,
        vis_graph: bal.vis_graph,
    }
}

pub fn add_noise(
    bal: BALProblem,
    translation_std: f64,
    rotation_std: f64,
    intrinsics_std: f64,
    point_std: f64,
    observations_std: f64,
) -> BALProblem {
    let n_translation = Normal::new(0.0, translation_std.into());
    let n_rotation = Normal::new(0.0, rotation_std.into());
    let n_intrinsics = Normal::new(0.0, intrinsics_std.into());
    let n_point = Normal::new(0.0, point_std.into());
    let n_observations = Normal::new(0.0, observations_std.into());
    let bal_std = bal.std().magnitude();

    let mut rng = rand::thread_rng();
    let cameras = bal
        .cameras
        .iter()
        .map(|c| {
            let dir = Basis3::from_axis_angle(unit_random(), Rad(n_rotation.sample(&mut rng)));
            let loc = unit_random() * bal_std * n_translation.sample(&mut rng) as f64;
            let intrin = unit_random() * n_intrinsics.sample(&mut rng) as f64;
            c.transform(dir, loc, intrin)
        })
        .collect();

    let points = bal
        .points
        .iter()
        .map(|p| p + unit_random() * n_point.sample(&mut rand::thread_rng()) as f64)
        .collect();

    let observations = bal
        .vis_graph
        .iter()
        .map(|obs| {
            obs.iter()
                .map(|(i, (x, y))| {
                    // random direction
                    let n = Normal::new(0.0, 1.0);
                    let nx = n.sample(&mut rand::thread_rng()) as f64;
                    let ny = n.sample(&mut rand::thread_rng()) as f64;
                    let m = (nx.powf(2.0) + ny.powf(2.0)).sqrt();
                    let r = n_observations.sample(&mut rand::thread_rng()) as f64;
                    let x_ = x + nx / m * r;
                    let y_ = y + ny / m * r;
                    (i.clone(), (x_, y_))
                })
                .collect::<Vec<(usize, (f64, f64))>>()
        })
        .collect();

    BALProblem {
        cameras: cameras,
        points: points,
        vis_graph: observations,
    }
}

/// Add incorrect correspondences by swapping two nearby observations.
pub fn add_incorrect_correspondences(bal: BALProblem, mismatch_chance: f64) -> BALProblem {
    let observations = bal
        .vis_graph
        .into_iter()
        .map(|mut obs| {
            if obs.len() > 1 {
                let mut rng = rand::thread_rng();
                for i in 0..obs.len() {
                    // Check if we should swap this entry
                    if rng.gen_range(0.0, 1.0) <= mismatch_chance {
                        // distance from this observation to all others
                        let mut dists = obs
                            .iter()
                            .map(|(_, (x, y))| {
                                (((obs[i].1).0 - x).powf(2.0) + ((obs[i].1).1 - y).powf(2.0)).sqrt()
                            })
                            .collect::<Vec<_>>();
                        // TODO: use max?
                        dists[i] = 10000000000.0;
                        let mut weights = dists.iter().map(|x| -x).collect::<Vec<_>>();
                        weights[i] = 0.0;
                        let m = weights.iter().fold(1. / 0., |a: f64, b: &f64| a.min(*b));
                        let weights = weights.iter().map(|x| x - m).collect::<Vec<_>>();
                        let j = WeightedIndex::new(weights).unwrap().sample(&mut rng);

                        // swap feature indices
                        let tmp = obs[i].0;
                        obs[i].0 = obs[j].0;
                        obs[j].0 = tmp;
                    }
                }
                obs
            } else {
                obs
            }
        })
        .collect();

    BALProblem {
        cameras: bal.cameras,
        points: bal.points,
        vis_graph: observations,
    }
}

pub fn drop_features(bal: BALProblem, drop_percent: f64) -> BALProblem {
    let mut rng = thread_rng();
    let observations = bal
        .vis_graph
        .iter()
        .map(|obs| {
            let l: usize = (obs.len() as f64 * drop_percent) as usize;
            let mut o = obs.clone();
            o.shuffle(&mut rng);
            o.truncate(l);
            o
        })
        .collect::<Vec<_>>();
    BALProblem {
        cameras: bal.cameras,
        points: bal.points,
        vis_graph: observations,
    }
}

pub fn split_landmarks(bal: BALProblem, split_percent: f64) -> BALProblem {
    let mut rng = thread_rng();
    // select which landmarks to split in two
    let l = bal.points.len();
    let n = (split_percent * l as f64) as usize;
    let inds = (0..l).choose_multiple(&mut rng, n);

    // copy split landmarks
    let mut points = bal.points.clone();
    points.extend(inds.iter().map(|i| bal.points[*i]));

    let split_inds: HashMap<usize, usize> = HashMap::from_iter(inds.into_iter().zip(l..(l + n)));

    // modify observations to sometimes view the new landmarks
    let mut observations = bal.vis_graph;
    for obs in observations.iter_mut() {
        for (i, _x) in obs.iter_mut() {
            if let Some(j) = split_inds.get(i) {
                // 50% chance to move this observation to the new landmark
                if rand::random() {
                    *i = *j;
                }
            }
        }
    }

    BALProblem {
        cameras: bal.cameras,
        points: points,
        vis_graph: observations,
    }
}

#[derive(Debug, PartialEq, Clone, Copy)]
struct IndexedVector3 {
    id: usize,
    p: Point3<f64>,
}

impl rstar::Point for IndexedVector3 {
    type Scalar = f64;
    const DIMENSIONS: usize = 3;

    fn generate(generator: impl Fn(usize) -> Self::Scalar) -> Self {
        IndexedVector3 {
            id: usize::max_value(),
            p: Point3::new(generator(0), generator(1), generator(2)),
        }
    }

    fn nth(&self, index: usize) -> Self::Scalar {
        match index {
            0 => self.p.x,
            1 => self.p.y,
            2 => self.p.z,
            _ => unreachable!(),
        }
    }

    fn nth_mut(&mut self, _index: usize) -> &mut Self::Scalar {
        unimplemented!()
    }
}

pub fn join_landmarks(bal: BALProblem, join_percent: f64) -> BALProblem {
    let observations = {
        let rtree = RTree::bulk_load(
            bal.points
                .iter()
                .enumerate()
                .map(|(i, x)| IndexedVector3 {
                    id: i,
                    p: x.clone(),
                })
                .collect(),
        );

        let mut rng = thread_rng();
        let l = bal.points.len();
        let n = (join_percent * l as f64) as usize;
        let inds = (0..bal.num_observations()).choose_multiple(&mut rng, n);
        // convert linear indices into camera, observation indices
        let obs_inds = inds
            .iter()
            .map(|i| {
                let mut j = 0;
                let mut c = 0;
                while bal.vis_graph[c].len() <= (i - j) {
                    j += bal.vis_graph[c].len();
                    c += 1;
                }
                (c, i - j)
            })
            .collect::<Vec<_>>();

        let mut observations = bal.vis_graph;
        for (c, i) in obs_inds {
            let pi = observations[c][i].0;

            // TODO: how to choose this constant
            let neighbor = rtree
                .nearest_neighbor_iter(&IndexedVector3 {
                    id: pi,
                    p: bal.points[pi],
                })
                .skip(1)
                .take(10)
                .choose(&mut rng)
                .expect("No neighbors?!");
            observations[c][i].0 = neighbor.id;
        }
        observations
    };

    BALProblem {
        cameras: bal.cameras,
        points: bal.points,
        vis_graph: observations,
    }
}
