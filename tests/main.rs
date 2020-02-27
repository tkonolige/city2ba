use assert_cmd::prelude::*;
use cgmath::Vector3;
use city2ba::noise::*;
use city2ba::synthetic::*;
use city2ba::*;
use predicates::prelude::*;
use std::path::*;
use std::process::Command;
use tempfile::tempdir;

#[test]
fn synthetic_blocks() -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempdir()?;

    let mut cmd = Command::cargo_bin("city2ba")?;
    cmd.arg("synthetic").arg(dir.path().join("blocks.bbal"));
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Bundle Adjustment Problem"));

    Ok(())
}

#[test]
fn test_bal_output() -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempdir()?;

    let mut cmd = Command::cargo_bin("city2ba")?;
    cmd.arg("synthetic").arg(dir.path().join("blocks.bal"));
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Bundle Adjustment Problem"));

    Ok(())
}

#[test]
fn noise_blocks() -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempdir()?;
    let bbal = dir.path().join("blocks.bbal");

    let mut cmd = Command::cargo_bin("city2ba")?;
    cmd.arg("synthetic").arg(bbal.clone());
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Bundle Adjustment Problem"));
    let mut cmd_noise = Command::cargo_bin("city2ba")?;
    cmd_noise
        .arg("noise")
        .arg(bbal)
        .arg(dir.path().join("blocks_noised.bbal"))
        .arg("--drift-strength")
        .arg("0.00001")
        .arg("--mismatch-chance")
        .arg("0.00001");
    cmd_noise
        .assert()
        .success()
        .stdout(predicate::str::contains("Initial error"))
        .stdout(predicate::str::contains("Final error"));

    Ok(())
}

#[test]
fn from_box_path() -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempdir()?;

    let mut cmd = Command::cargo_bin("city2ba")?;
    cmd.arg("generate")
        .arg(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/box.obj"))
        .arg(dir.path().join("box.bal"))
        .arg("--cameras")
        .arg("100")
        .arg("--points")
        .arg("100")
        .arg("--path")
        .arg("BezierCurve");
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Total reprojection error"));

    Ok(())
}

#[test]
fn from_box_path_step() -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempdir()?;

    let mut cmd = Command::cargo_bin("city2ba")?;
    cmd.arg("generate")
        .arg(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/box.obj"))
        .arg(dir.path().join("box.bal"))
        .arg("--cameras")
        .arg("100")
        .arg("--points")
        .arg("100")
        .arg("--path")
        .arg("BezierCurve")
        .arg("--step-size")
        .arg("0.1");
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Total reprojection error"));

    Ok(())
}

#[test]
fn from_box_path_ground() -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempdir()?;

    let mut cmd = Command::cargo_bin("city2ba")?;
    cmd.arg("generate")
        .arg(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/box.obj"))
        .arg(dir.path().join("box.bal"))
        .arg("--cameras")
        .arg("100")
        .arg("--points")
        .arg("100")
        .arg("--ground")
        .arg("-1.0");
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Total reprojection error"));

    Ok(())
}

fn test_grid() -> BAProblem<SnavelyCamera> {
    synthetic_grid(10, 20, 3, 5., 1., 1., 1., 10., false)
}

#[test]
fn normalized_drift() {
    let ba = test_grid();
    let err_start = ba.total_reprojection_error(2.0);
    let ba = add_drift_normalized(ba, 0.1, 0.1, 0.1);
    let err_end = ba.total_reprojection_error(2.0);
    assert!(err_end > err_start);
}

#[test]
fn noise() {
    let ba = test_grid();
    let err_start = ba.total_reprojection_error(2.0);
    let ba = add_noise(ba, 0.1, 0.1, 0.1, 0.1);
    let err_end = ba.total_reprojection_error(2.0);
    assert!(err_end > err_start);
}

#[test]
fn incorrect_correspondences() {
    let ba = test_grid();
    let err_start = ba.total_reprojection_error(2.0);
    let ba = add_incorrect_correspondences(ba, 0.01);
    let err_end = ba.total_reprojection_error(2.0);
    assert!(err_end > err_start);
}

#[test]
fn test_drop_features() {
    let ba = test_grid();
    let err_start = ba.total_reprojection_error(2.0);
    let ba = drop_features(ba, 0.1);
    let err_end = ba.total_reprojection_error(2.0);
    assert!(err_end >= err_start);
}

#[test]
fn test_split_landmarks() {
    let ba = test_grid();
    let err_start = ba.total_reprojection_error(2.0);
    let ba = split_landmarks(ba, 0.1);
    let err_end = ba.total_reprojection_error(2.0);
    assert!(err_end >= err_start);
}

#[test]
fn test_join_landmarks() {
    let ba = test_grid();
    let err_start = ba.total_reprojection_error(2.0);
    let ba = join_landmarks(ba, 0.01);
    let err_end = ba.total_reprojection_error(2.0);
    assert!(err_end > err_start);
}

#[test]
fn sin_noise() {
    let ba = test_grid();
    let err_start = ba.total_reprojection_error(2.0);
    let ba = add_sin_noise(ba, Vector3::new(1.0, 1.0, 0.0), Vector3::unit_y(), 1., 2.);
    let err_end = ba.total_reprojection_error(2.0);
    assert!(err_end > err_start);
}

#[test]
fn test_line() {
    let ba = synthetic_line::<SnavelyCamera>(30, 40, 10., 1., 1., 1., 10., false);
    assert!(ba.num_cameras() > 20, "num_cameras: {}", ba.num_cameras());
}
