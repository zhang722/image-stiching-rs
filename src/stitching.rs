use std::{error::Error, sync::Arc};

use image::{GenericImageView, GenericImage};
use nalgebra as na;

use crate::{
    filter,
    harris,
    homography,
    ransac,
};

pub fn bilinear_interpolation(x: f64, y: f64, img: &image::DynamicImage) -> image::Rgba<u8> {
    let (width, height) = img.dimensions();

    // if x < 0.0 || y < 0.0 || x > width as f64 || y > height as f64{
    //     return image::Rgba([0, 0, 0, 255]);
    // }

    let x0 = x.floor() as u32;
    let x1 = x.ceil() as u32;
    let y0 = y.floor() as u32;
    let y1 = y.ceil() as u32;
    let x0 = x0.min(width - 1);
    let x1 = x1.min(width - 1);
    let y0 = y0.min(height - 1);
    let y1 = y1.min(height - 1);
    let p00 = img.get_pixel(x0, y0);
    let p01 = img.get_pixel(x0, y1);
    let p10 = img.get_pixel(x1, y0);
    let p11 = img.get_pixel(x1, y1);
    let p0 = image::Rgb([
        (p00[0] as f64 * (x1 as f64 - x) + p10[0] as f64 * (x - x0 as f64)) as u8,
        (p00[1] as f64 * (x1 as f64 - x) + p10[1] as f64 * (x - x0 as f64)) as u8,
        (p00[2] as f64 * (x1 as f64 - x) + p10[2] as f64 * (x - x0 as f64)) as u8,
    ]);
    let p1 = image::Rgb([
        (p01[0] as f64 * (x1 as f64 - x) + p11[0] as f64 * (x - x0 as f64)) as u8,
        (p01[1] as f64 * (x1 as f64 - x) + p11[1] as f64 * (x - x0 as f64)) as u8,
        (p01[2] as f64 * (x1 as f64 - x) + p11[2] as f64 * (x - x0 as f64)) as u8,
    ]);
    image::Rgba([
        (p0[0] as f64 * (y1 as f64 - y) + p1[0] as f64 * (y - y0 as f64)) as u8,
        (p0[1] as f64 * (y1 as f64 - y) + p1[1] as f64 * (y - y0 as f64)) as u8,
        (p0[2] as f64 * (y1 as f64 - y) + p1[2] as f64 * (y - y0 as f64)) as u8,
        255,
    ])
}

pub fn get_vertexes(img: &image::DynamicImage) -> Vec<na::Point2<f64>> {
    let (width, height) = img.dimensions();
    vec![
        na::Point2::<f64>::new(0.0, 0.0),
        na::Point2::<f64>::new(width as f64, 0.0),
        na::Point2::<f64>::new(0.0, height as f64),
        na::Point2::<f64>::new(width as f64, height as f64),
    ]
}

fn is_in_image(x: f64, y: f64, img: &image::DynamicImage) -> bool {
    let (width, height) = img.dimensions();

    x >= 0.0 && y >= 0.0 && x + 1.0 < width as f64 && y + 1.0 < height as f64 
}

fn get_mix_weight(x: f64, y: f64, img: &image::DynamicImage) -> f64 {
    let (width, height) = img.dimensions();
    assert!(x >= 0.0 && y >= 0.0 && x + 1.0 < width as f64 && y + 1.0 < height as f64);
    let min_dis = x
        // .min(y)
        .min(width as f64 - x)
        // .min(height as f64 - y)
        ;
    if min_dis > 20.0 {
        return 1.0;
    }
    min_dis / 20.0
}

pub fn stitching_two_imgs(keeped_img: image::DynamicImage, stitched_img: image::DynamicImage) -> Result<image::DynamicImage, Box<dyn Error>> {

    let gray1 = filter::filter(&keeped_img, &filter::kernel::GaussianKernel::new(3, 2.0));
    let gray2 = filter::filter(&stitched_img, &filter::kernel::GaussianKernel::new(3, 2.0));

    let corners1 = harris::detect_harris_corners(&keeped_img, harris::HarrisCornerDetector::Harris, Some(0.04), 2.0, 2.0, 1.0, 500);
    let pairs = harris::generate_pairs(256, 9, &mut rand::thread_rng());
    let descriptors = harris::brief_descriptor(&gray1, &corners1, 9, &pairs);
    let corners2 = harris::detect_harris_corners(&stitched_img, harris::HarrisCornerDetector::Harris, Some(0.04), 2.0, 2.0, 1.0, 500);
    let descriptors2 = harris::brief_descriptor(&gray2, &corners2, 9, &pairs);

    let matches = harris::match_descriptors(&descriptors, &descriptors2, 0.1);

    // in HarrisMatch, first is src, second is des
    // that is: second = H * first
    let h = match ransac::ransac::<homography::HomographyModel>(&matches, 5, 1000, 2.0, matches.len() / 2) {
        Some(h) => h,
        None => return Err("no homography found".into()),
    };
    
    let vertexes = get_vertexes(&stitched_img);
    
    let h_inv = match h.try_inverse() {
        Some(h) => h,
        None => return Err("no inverse homography found".into()),
    };

    let mut vertexes = vertexes.iter().map(|v| {
        let v = h_inv * na::Vector3::<f64>::new(v.x, v.y, 1.0);
        na::Point2::<f64>::new(v.x / v.z, v.y / v.z)
    }).collect::<Vec<_>>();

    let (width, height) = keeped_img.dimensions();
    vertexes.push(na::Point2::<f64>::new(0.0, 0.0));
    vertexes.push(na::Point2::<f64>::new(width as f64, 0.0));
    vertexes.push(na::Point2::<f64>::new(0.0, height as f64));
    vertexes.push(na::Point2::<f64>::new(width as f64, height as f64));
    for v in vertexes.iter() {
        println!("v: ({}, {})", v.x, v.y);
    }

    let offset_x = match vertexes.iter().map(|v| v.x).reduce(f64::min) {
        Some(n) if n < 0.0 => {-n},
        _ => {0.0},
    };

    let offset_y = match vertexes.iter().map(|v| v.y).reduce(f64::min) {
        Some(n) if n < 0.0 => {-n},
        _ => {0.0},
    };

    // get max x of vertexes
    let max_x = vertexes.iter().map(|v| v.x).reduce(f64::max).unwrap();
    let max_y = vertexes.iter().map(|v| v.y).reduce(f64::max).unwrap();

    // new empty image
    let mut new_img = image::DynamicImage::new_rgb8((max_x + offset_x) as u32, (max_y + offset_y) as u32);

    // copy keeped_img to new_img
    image::imageops::overlay(&mut new_img, &keeped_img, offset_x as i64, offset_y as i64);

    // find corresponding point in stitching_img
    for x in 0..new_img.width() {
        for y in 0..new_img.height() {
            let v = h * na::Vector3::<f64>::new(x as f64 - offset_x, y as f64 - offset_y, 1.0);
            let projed_x = v.x / v.z;
            let projed_y = v.y / v.z;
            if is_in_image(projed_x, projed_y, &stitched_img) {
                let mut p = bilinear_interpolation(projed_x, projed_y, &stitched_img);

                if is_in_image(x as f64 - offset_x, y as f64 - offset_y, &keeped_img) {
                    let keeped_p = keeped_img.get_pixel((x as f64 - offset_x) as u32, (y as f64 - offset_y) as u32);
                    let weight = get_mix_weight(projed_x, projed_y, &stitched_img);
                    p[0] = (weight * p[0] as f64 + (1.0 - weight) * keeped_p[0] as f64) as u8;
                    p[1] = (weight * p[1] as f64 + (1.0 - weight) * keeped_p[1] as f64) as u8;
                    p[2] = (weight * p[2] as f64 + (1.0 - weight) * keeped_p[2] as f64) as u8;
                }

                new_img.put_pixel(x, y, p);
            }
        }
    }

    Ok(new_img)
}


#[cfg(test)]
mod test{
    use std::error::Error;

    #[test]
    fn test_stitching_two_imgs() -> Result<(), Box<dyn Error>> {
        use image::io::Reader as ImageReader;

        let img1_path = "hw/IMG_0627.JPG";
        let img2_path = "hw/IMG_0628.JPG";
        let img1 = ImageReader::open(img1_path)?.decode()?;
        let img2 = ImageReader::open(img2_path)?.decode()?;

        let new_img = super::stitching_two_imgs(img1, img2)?;
        new_img.save("stitched.jpg")?;

        Err("end".into())
    }
    #[test]
    fn test_stitching_multi_imgs() -> Result<(), Box<dyn Error>> {
        use image::io::Reader as ImageReader;
        let keeped_img_path = "hw/IMG_0627.JPG";
        let stitched_img_paths = (28..=32).into_iter().map(|f| format!("hw/IMG_06{:02}.JPG", f)).collect::<Vec<_>>();
        let mut keeped_img = ImageReader::open(keeped_img_path)?.decode()?;

        for stitched_img_path in stitched_img_paths {
            let stitched_img = ImageReader::open(stitched_img_path)?.decode()?;
            let new_img = super::stitching_two_imgs(keeped_img.clone(), stitched_img)?;
            keeped_img = new_img;
        }
        keeped_img.save("stitched.jpg")?;

        Ok(())
        // Err("end".into())
    }
}