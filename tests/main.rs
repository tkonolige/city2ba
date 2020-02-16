use assert_cmd::prelude::*;
use predicates::prelude::*;
use std::path::*;
use std::process::Command;
use tempfile::tempdir;

#[test]
fn synthetic_blocks() -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempdir()?;

    let mut cmd = Command::cargo_bin("synthetic")?;
    cmd.arg(dir.path().join("blocks.bbal"));
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Bundle Adjustment Problem"));

    Ok(())
}

#[test]
fn test_bal_output() -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempdir()?;

    let mut cmd = Command::cargo_bin("synthetic")?;
    cmd.arg(dir.path().join("blocks.bal"));
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Bundle Adjustment Problem"));

    Ok(())
}

#[test]
fn noise_blocks() -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempdir()?;
    let bbal = dir.path().join("blocks.bbal");

    let mut cmd = Command::cargo_bin("synthetic")?;
    cmd.arg(bbal.clone());
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Bundle Adjustment Problem"));
    let mut cmd_noise = Command::cargo_bin("noise")?;
    cmd_noise
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
    cmd.arg(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/box.obj"))
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
    cmd.arg(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/box.obj"))
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
    cmd.arg(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/box.obj"))
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
