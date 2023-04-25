use std::{error::Error, sync::Arc};

use image::GenericImageView;
use nalgebra as na;

use crate::{
    filter,
    harris,
    homography,
    ransac,
};

pub fn get_vertexes(img: &image::DynamicImage) -> Vec<na::Point2<f64>> {
    let (width, height) = img.dimensions();
    let mut vertexes = Vec::new();
    vertexes.push(na::Point2::<f64>::new(0.0, 0.0));
    vertexes.push(na::Point2::<f64>::new(width as f64, 0.0));
    vertexes.push(na::Point2::<f64>::new(0.0, height as f64));
    vertexes.push(na::Point2::<f64>::new(width as f64, height as f64));
    vertexes
}

pub fn stitching_two_imgs(keeped_img: image::DynamicImage, stitched_img: image::DynamicImage) -> Result<image::DynamicImage, Box<dyn Error>> {

    let gray1 = filter::filter(&keeped_img, &filter::kernel::GaussianKernel::new(3, 2.0));
    let gray2 = filter::filter(&stitched_img, &filter::kernel::GaussianKernel::new(3, 2.0));

    let corners1 = harris::detect_harris_corners(&keeped_img, harris::HarrisCornerDetector::Harris, Some(0.04), 2.0, 2.0, 1.0, 500);
    let pairs = harris::generate_pairs(256, 9, &mut rand::thread_rng());
    let descriptors = harris::brief_descriptor(&gray1, &corners1, 9, &pairs);
    let corners2 = harris::detect_harris_corners(&stitched_img, harris::HarrisCornerDetector::Harris, Some(0.04), 2.0, 2.0, 1.0, 500);
    let descriptors2 = harris::brief_descriptor(&gray2, &corners2, 9, &pairs);

    let matches = harris::match_descriptors(&descriptors2, &descriptors, 0.1);

    // in HarrisMatch, first is src, second is des
    // that is: second = H * first
    let h = match ransac::ransac::<homography::HomographyModel>(&matches, 5, 1000, 3.0, matches.len() / 2) {
        Some(h) => h,
        None => return Err("no homography found".into()),
    };
    
    let mut vertexes = get_vertexes(&stitched_img);
    let (width, height) = keeped_img.dimensions();
    vertexes.push(na::Point2::<f64>::new(0.0, 0.0));
    vertexes.push(na::Point2::<f64>::new(width as f64, 0.0));
    vertexes.push(na::Point2::<f64>::new(0.0, height as f64));
    vertexes.push(na::Point2::<f64>::new(width as f64, height as f64));
    
    let vertexes = vertexes.iter().take(4).map(|v| {
        let v = h * na::Vector3::<f64>::new(v.x, v.y, 1.0);
        na::Point2::<f64>::new(v.x / v.z, v.y / v.z)
    }).collect::<Vec<_>>();

    let offset_x = match vertexes.iter().map(|v| v.x).reduce(f64::min) {
        Some(n) if n < 0.0 => {-n},
        _ => {0.0},
    };

    let offset_y = match vertexes.iter().map(|v| v.y).reduce(f64::min) {
        Some(n) if n < 0.0 => {-n},
        _ => {0.0},
    };



}