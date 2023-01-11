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

# Statement of Need

Synthetic datasets have zero error and so are useful in testing the accuracy and correctness of different methods.
To test different algorithms, the ground truth datasets need some form of added error.
Currently, there's a lack of tools for adding errors to simulate different inaccuracies in Structure from Motion (SfM) methods. 
**City2BA** aims to address this need by providing variety of different types of error.

# Summary

Bundle adjustment is a global nonlinear optimization step used in SfM and simultaneous localization and mapping (SLAM).
It is usually formulated as a nonlinear least-squares problem where the goal is to minimize the error between the projected location of 3D points in each camera and the actual observed location of the point in the camera frames.
@triggs1999bundle provides a good overview of the formulation and uses of bundle adjustment.
For SLAM, the bundle adjustment problem is small, but for SfM, problem sizes can grow very large.
Ideally, developers of new bundle adjustment algorithms would like to test against real world data however not many datasets are available (the authors only know of [1DSFM](http://www.cs.cornell.edu/projects/1dsfm/) [@wilson2014robust] and [Bundle Adjustment in the Large](http://grail.cs.washington.edu/projects/bal/) [@agarwal2010bundle]), and these datasets are limited in size and structure.
We know large datasets exists and are in use (see @klingner2013street), but these datasets are not available to the public.
To facilitate the development of bundle adjustment algorithms, we developed the **City2BA** package which can generate large, synthetic bundle adjustment datasets.

**City2BA** provides two main features: generation of synthetic datasets from models, and tools to add noise to existing datasets.
Synthetic dataset generation can either use an existing 3D model (in `.obj` format), or can operate on an implicit model of a grid or a line.
The user can specify the problem size as well as parameters like number of 3D points and maximum distance between a camera and an observed point.
The user can also modify problem structure by using different 3D models or placing different camera paths.
Cameras can either be generated in a streetview-like scenario where cameras are placed along a path, or placed in random locations in the geometry.

**City2BA** is provided both as a set of command line tools (for convenience) and as a library (for extensibility).
Currently, **City2BA** ships with a single camera model commonly used in the literature (see @agarwal2010bundle), but the user can add their own camera model if they so choose.
It is optimized and can generate very large models (100,000 cameras, 1,000,000) in less than an hour.

# References
