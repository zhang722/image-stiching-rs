use std::error::Error;

use image::{Pixel, GenericImage, GenericImageView};
use nalgebra as na;

use crate::{
    harris,
    ransac,
};


fn Normalize(point_vec: &Vec<na::Point2<f64>>) 
-> Result<(Vec<na::Point2<f64>>, na::Matrix3<f64>), Box<dyn Error>>
{
    let mut norm_T = na::Matrix3::<f64>::identity();
    let mut normed_point_vec = Vec::new();
    let mut mean_x = 0.0;
    let mut mean_y = 0.0;
    for p in point_vec {
        mean_x += p.x;
        mean_y += p.y;
    }
    mean_x /= point_vec.len() as f64;
    mean_y /= point_vec.len() as f64;
    let mut mean_dev_x = 0.0;
    let mut mean_dev_y = 0.0;
    for p in point_vec {
        mean_dev_x += (p.x - mean_x).abs();
        mean_dev_y += (p.y - mean_y).abs();
    }
    mean_dev_x /= point_vec.len() as f64;
    mean_dev_y /= point_vec.len() as f64;
    let sx = 1.0 / mean_dev_x;
    let sy = 1.0 / mean_dev_y;

    for p in point_vec {
        let mut p_tmp = na::Point2::<f64>::new(0.0, 0.0);
        p_tmp.x = sx * p.x - mean_x * sx;
        p_tmp.y = sy * p.y - mean_y * sy;
        normed_point_vec.push(p_tmp);
    }
    norm_T[(0, 0)] = sx;
    norm_T[(0, 2)] = -mean_x * sx;
    norm_T[(1, 1)] = sy;
    norm_T[(1, 2)] = -mean_y * sy;

    Ok((normed_point_vec, norm_T))
}


pub fn compute_h(img_points: &Vec<na::Point2<f64>>, world_points: &Vec<na::Point2<f64>>) -> Result<na::Matrix3::<f64>, Box<dyn Error>> {
    let num_points = img_points.len();
    assert_eq!(num_points, world_points.len());

    // at least 4 point if want to compute H
    assert!(num_points > 3);

    type MatrixXx9<T> = na::Matrix<T, na::Dyn, na::U9, na::VecStorage<T, na::Dyn, na::U9>>;
    type RowVector9<T> = na::Matrix<T, na::U1, na::U9, na::ArrayStorage<T, 1, 9>>;

    let norm_img = Normalize(img_points)?;
    let norm_world = Normalize(world_points)?;

    let mut a = MatrixXx9::<f64>::zeros(num_points * 2);

    let img_world_points_iter = norm_img.0.iter().zip(norm_world.0.iter());
    for (idx, (img_point, world_point)) in img_world_points_iter.enumerate() {
        let u = img_point.x;
        let v = img_point.y;
        let x_w = world_point.x;
        let y_w = world_point.y;

        let ax = RowVector9::<f64>::from_vec(vec![
            x_w, y_w, 1.0, 0.0, 0.0, 0.0, -u*x_w, -u*y_w, -u 
        ]);
        let ay = RowVector9::<f64>::from_vec(vec![
            0.0, 0.0, 0.0, x_w, y_w, 1.0, -v*x_w, -v*y_w, -v
        ]);
        
        a.set_row(2 * idx, &ax);
        a.set_row(2 * idx + 1, &ay);
    } 
    let svd = a.svd(true, true);
    let v_t = match svd.v_t {
        Some(v_t) => v_t,
        None => return Err(From::from("compute V failed")),
    };
    let last_row = v_t.row(v_t.nrows() - 1);

    // construct matrix from vector
    let mut ret = na::Matrix3::<f64>::from_iterator(last_row.into_iter().cloned()).transpose();


    ret = match norm_img.1.try_inverse() {
        Some(m) => m,
        None => return Err(From::from("compute inverse norm_img failed")),
    } * ret * norm_world.1;

    ret = ret / ret[(2, 2)];
    
    Ok(ret)  
}

struct HomographyModel;

impl ransac::Model for HomographyModel {
    type Point = harris::HarrisMatch;
    type ModelParams = na::Matrix3<f64>;

    fn estimate_model(points: &[Self::Point]) -> Option<Self::ModelParams> {
        let src = points.iter().map(|p| na::Point2::<f64>::new(p.first.x as f64, p.first.y as f64)).collect::<Vec<_>>();
        let des = points.iter().map(|p| na::Point2::<f64>::new(p.second.x as f64, p.second.y as f64)).collect::<Vec<_>>();

        match crate::homography::compute_h(&des, &src) {
            Ok(h) => Some(h),
            Err(e) => {
                println!("estimate model failed:{}", e);
                None
            },
        }

    }

    fn consensus_distance(params: &Self::ModelParams, point: &Self::Point) -> f64 {
        let des = point.second;
        let src = point.first;
        let src = na::Point3::<f64>::new(src.x as f64, src.y as f64, 1.0);
        let des = na::Point3::<f64>::new(des.x as f64, des.y as f64, 1.0);
        let projected_src = params * src;
        let projected_src = projected_src / projected_src[2];

        (projected_src - des).norm()
    }
}


#[test]
fn test_homography() -> Result<(), Box<dyn Error>> {
    use crate::filter;
    use image::io::Reader as ImageReader;
    use crate::harris::*;

    let img1_path = "hw/IMG_0627.JPG";
    let img2_path = "hw/IMG_0628.JPG";
    let img1 = ImageReader::open(img1_path)?.decode()?;
    let img2 = ImageReader::open(img2_path)?.decode()?;
    let gray1 = filter::filter(&img1, &filter::kernel::GaussianKernel::new(3, 2.0));
    let gray2 = filter::filter(&img2, &filter::kernel::GaussianKernel::new(3, 2.0));

    let corners1 = detect_harris_corners(&img1, HarrisCornerDetector::Harris, Some(0.04), 2.0, 2.0, 1.0, 500);
    let pairs = generate_pairs(256, 9, &mut rand::thread_rng());
    let descriptors = brief_descriptor(&gray1, &corners1, 9, &pairs);
    let corners2 = detect_harris_corners(&img2, HarrisCornerDetector::Harris, Some(0.04), 2.0, 2.0, 1.0, 500);
    let descriptors2 = brief_descriptor(&gray2, &corners2, 9, &pairs);


    let matches = match_descriptors(&descriptors, &descriptors2, 0.1);
    let des = matches.iter().map(|p| na::Point2::<f64>::new(p.first.x as f64, p.first.y as f64)).collect::<Vec<_>>();
    let src = matches.iter().map(|p| na::Point2::<f64>::new(p.second.x as f64, p.second.y as f64)).collect::<Vec<_>>();

    // let h = compute_h(&src, &des)?;

    let h = match ransac::ransac::<HomographyModel>(&matches, 5, 100, 3.0, matches.len() / 2) {
        Some(h) => h,
        None => return Err("no homography found".into()),
    };
    let project = |x: u32, y: u32| -> (u32, u32) {
        let mut p = na::Matrix3x1::<f64>::from_column_slice(&[x as f64, y as f64, 1.0]);
        p = h * p;
        p = p / p[(2, 0)];
        (p[(0, 0)] as u32, p[(1, 0)] as u32)
    };

    let project_img = |img1: image::DynamicImage| -> image::DynamicImage {
        let mut img = image::DynamicImage::new_rgb8(img1.width(), img1.height());
        for x in 0..img1.width() {
            for y in 0..img1.height() {
                let (projed_x, projed_y) = project(x, y);
                if projed_x < img1.width() && projed_y < img1.height() {
                    let pixel = img1.get_pixel(x, y);
                    img.put_pixel(projed_x, projed_y, pixel);
                }
            }
        }
        img
    };

    let projected_img = project_img(img1);
    projected_img.save("projected_img.png")?;

    println!("{}", h);
    assert!(1==2);
    Ok(())
}


#[test]
fn test_opencv_findhomography() -> Result<(), Box<dyn Error>> {
    use crate::filter;
    use image::io::Reader as ImageReader;
    use crate::harris::*;
    use opencv::core::*;
    use opencv::imgcodecs::*;
    use opencv::imgproc::*;
    use opencv::features2d::*;
    use opencv::calib3d::*;
    use opencv::types::*;

    let img1_path = "hw/IMG_0627.JPG";
    let img2_path = "hw/IMG_0628.JPG";
    let img1 = ImageReader::open(img1_path)?.decode()?;
    let img2 = ImageReader::open(img2_path)?.decode()?;
    let gray1 = filter::filter(&img1, &filter::kernel::GaussianKernel::new(3, 2.0));
    let gray2 = filter::filter(&img2, &filter::kernel::GaussianKernel::new(3, 2.0));

    let corners1 = detect_harris_corners(&img1, HarrisCornerDetector::Harris, Some(0.04), 2.0, 2.0, 1.0, 500);
    let pairs = generate_pairs(256, 9, &mut rand::thread_rng());
    let descriptors = brief_descriptor(&gray1, &corners1, 9, &pairs);
    let corners2 = detect_harris_corners(&img2, HarrisCornerDetector::Harris, Some(0.04), 2.0, 2.0, 1.0, 500);
    let descriptors2 = brief_descriptor(&gray2, &corners2, 9, &pairs);
    // opencv findhomograpy of two images
    let matches = match_descriptors(&descriptors, &descriptors2, 0.1);
    let src = matches.iter().map(|p| na::Point2::<f64>::new(p.first.x as f64, p.first.y as f64)).collect::<Vec<_>>();
    let des = matches.iter().map(|p| na::Point2::<f64>::new(p.second.x as f64, p.second.y as f64)).collect::<Vec<_>>();
    let src = src.iter().map(|p| [p.x, p.y]).collect::<Vec<_>>();
    let des = des.iter().map(|p| [p.x, p.y]).collect::<Vec<_>>();
    let src_mat = Mat::from_slice_2d(&src).unwrap();
    let des_mat = Mat::from_slice_2d(&des).unwrap();
    let mut h = Mat::default();
    let homo = find_homography(&src_mat, &des_mat, &mut h, RANSAC, 3.0)?;

    // Mat to na::Matrix3
    let mut h = na::Matrix3::<f64>::zeros();
    for i in 0..3i32 {
        for j in 0..3i32 {
            h[(i as usize, j as usize)] = *homo.at_2d::<f64>(i, j)?;
        }
    }
    let project = |x: u32, y: u32| -> (u32, u32) {
        let mut p = na::Matrix3x1::<f64>::from_column_slice(&[x as f64, y as f64, 1.0]);
        p = h * p;
        p = p / p[(2, 0)];
        (p[(0, 0)] as u32, p[(1, 0)] as u32)
    };

    let project_img = |img1: image::DynamicImage| -> image::DynamicImage {
        let mut img = image::DynamicImage::new_rgb8(img1.width(), img1.height());
        for x in 0..img1.width() {
            for y in 0..img1.height() {
                let (projed_x, projed_y) = project(x, y);
                if projed_x < img1.width() && projed_y < img1.height() {
                    let pixel = img1.get_pixel(x, y);
                    img.put_pixel(projed_x, projed_y, pixel);
                }
            }
        }
        img
    };

    let projected_img = project_img(img1);
    projected_img.save("projected_img.png")?;

    assert!(1==2);
    Ok(())
}