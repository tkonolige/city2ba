extern crate cgmath;
extern crate ply_rs;
extern crate structopt;

use ply_rs::ply::{
    Addable, DefaultElement, ElementDef, Ply, Property, PropertyDef, PropertyType, ScalarType,
};
use ply_rs::writer::Writer;

use structopt::StructOpt;

use city2bal::*;

use cgmath::Point3;

use std::fs::File;
use std::io::BufWriter;

#[derive(StructOpt, Debug)]
#[structopt(name = "basic")]
struct Opt {
    #[structopt(name = "FILE", parse(from_os_str))]
    input: std::path::PathBuf,

    #[structopt(name = "OUT", parse(from_os_str))]
    out: std::path::PathBuf,
}

/// Write camera locations out to a ply file. Does not provide color or orientation.
fn write_cameras(
    path: &std::path::Path,
    cameras: &Vec<Camera>,
    points: &Vec<Point3<f64>>,
) -> Result<(), std::io::Error> {
    let mut ply = Ply::<DefaultElement>::new();
    let mut point_element = ElementDef::new("vertex".to_string());
    let p = PropertyDef::new("x".to_string(), PropertyType::Scalar(ScalarType::Float));
    point_element.properties.add(p);
    let p = PropertyDef::new("y".to_string(), PropertyType::Scalar(ScalarType::Float));
    point_element.properties.add(p);
    let p = PropertyDef::new("z".to_string(), PropertyType::Scalar(ScalarType::Float));
    point_element.properties.add(p);
    let p = PropertyDef::new("red".to_string(), PropertyType::Scalar(ScalarType::UChar));
    point_element.properties.add(p);
    let p = PropertyDef::new("green".to_string(), PropertyType::Scalar(ScalarType::UChar));
    point_element.properties.add(p);
    let p = PropertyDef::new("blue".to_string(), PropertyType::Scalar(ScalarType::UChar));
    point_element.properties.add(p);
    ply.header.elements.add(point_element);

    // Add first point
    let mut cs: Vec<_> = cameras
        .iter()
        .map(|camera| {
            let mut point = DefaultElement::new();
            point.insert("x".to_string(), Property::Float(camera.center()[0] as f32));
            point.insert("y".to_string(), Property::Float(camera.center()[1] as f32));
            point.insert("z".to_string(), Property::Float(camera.center()[2] as f32));
            point.insert("red".to_string(), Property::UChar(255));
            point.insert("green".to_string(), Property::UChar(0));
            point.insert("blue".to_string(), Property::UChar(0));
            point
        })
        .collect();

    let pts = points.iter().map(|point| {
        let mut p = DefaultElement::new();
        p.insert("x".to_string(), Property::Float(point[0] as f32));
        p.insert("y".to_string(), Property::Float(point[1] as f32));
        p.insert("z".to_string(), Property::Float(point[2] as f32));
        p.insert("red".to_string(), Property::UChar(0));
        p.insert("green".to_string(), Property::UChar(255));
        p.insert("blue".to_string(), Property::UChar(0));
        p
    });

    cs.extend(pts);

    ply.payload.insert("vertex".to_string(), cs);

    let mut file = BufWriter::new(File::create(path)?);
    let writer = Writer::new();
    writer.write_ply(&mut file, &mut ply).map(|_| ())
}

fn main() -> std::result::Result<(), city2bal::Error> {
    let opt = Opt::from_args();

    let bal = BALProblem::from_file(&opt.input)?;

    write_cameras(&opt.out, &bal.cameras, &bal.points)?;

    Ok(())
}
