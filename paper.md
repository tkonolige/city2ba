---
title: 'City2BA: Tools for creating synthetic bundle adjustment problems'
tags:
  - rust
  - bundle adjustment
  - structure from motion
authors:
  - name: Tristan Konolige
    orcid: 0000-0002-5052-6479
    affiliation: 1
affiliations:
 - name: University of Colorado Boulder
   index: 1
date: 27 February 2020
bibliography: paper.bib
---

# City2BA

Bundle adjustment is a global nonlinear optimization step used in structure from motion (SfM) and simultaneous localization and mapping (SLAM).
Bundle adjustment is usually formulated as a nonlinear least-squares problem where the goal is to minimize the error between the projected location of 3D points into each camera and the actual observed location of the point in the camera frames.
@triggs1999bundle provides a good overview of the formulation and uses of bundle adjustment.
For SLAM problems the bundle adjustment problem is relatively small, but for SfM, problems sizes can grow very large.
Ideally, developers of new bundle adjustment algorithms would like to test against real world data.
However, not many datasets are available (the authors only know of [1DSFM]: http://www.cs.cornell.edu/projects/1dsfm/ and [Bundle Adjustment in the Large]: http://grail.cs.washington.edu/projects/bal/), and these datasets are limited in size and structure.
To facilitate the development of bundle adjustment algorithms, we developed the *City2BA* package which can generate large, realistic bundle adjustment datasets.

*City2BA* provides two main features: generation of synthetic datasets from models, and tools to add noise to existing datasets.
Synthetic dataset generation can either use an existing 3D model (in `.obj` format), or can operate on an implicit model of a grid or a line.
The user can specify number of cameras and points, maximum distance between cameras and points, and the camera model itself.
Cameras can either be generated in a streetview-like scenario where cameras are placed along a path, or placed in random locations in the geometry.
Synthetic datasets have zero error and so are useful in testing the accuracy and correctness of different methods.
To test different algorithms, the ground truth datasets need some form of added error.
The tools we provide can add a variety of different types of error to simulate different inaccuracies in SfM methods.
For example, we provide long range drifting effects that mimic error from accelerometer noise. 
Currently, *City2BA* ships with a single camera model commonly used in the literature (see @agarwal2010bundle).
However, our library allows the user to add their own camera model if they so choose.

# References
