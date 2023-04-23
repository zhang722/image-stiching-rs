use std::error::Error;
use image::{Pixel, GenericImage, GenericImageView};
use nalgebra as na;

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
    for p in img_points {
        println!("img_points: ({}, {})", p.x, p.y);
    }

    type MatrixXx9<T> = na::Matrix<T, na::Dyn, na::U9, na::VecStorage<T, na::Dyn, na::U9>>;
    type RowVector9<T> = na::Matrix<T, na::U1, na::U9, na::ArrayStorage<T, 1, 9>>;

    let norm_img = Normalize(img_points)?;
    let norm_world = Normalize(world_points)?;
    for p in &norm_img.0 {
        println!("img_points: ({}, {})", p.x, p.y);
    }

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
    let last_row = v_t.row(8);

    // construct matrix from vector
    let mut ret = na::Matrix3::<f64>::from_iterator(last_row.into_iter().cloned()).transpose();


    ret = match norm_img.1.try_inverse() {
        Some(m) => m,
        None => return Err(From::from("compute inverse norm_img failed")),
    } * ret * norm_world.1;

    ret = ret / ret[(2, 2)];
    
    Ok(ret)  
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
    let src = matches.iter().map(|p| na::Point2::<f64>::new(p.first.x as f64, p.first.y as f64)).collect::<Vec<_>>();
    let des = matches.iter().map(|p| na::Point2::<f64>::new(p.second.x as f64, p.second.y as f64)).collect::<Vec<_>>();

    let h = compute_h(&des, &src)?;
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
                let (x, y) = project(x, y);
                if x < img1.width() && y < img1.height() {
                    let pixel = img1.get_pixel(x, y);
                    img.put_pixel(x, y, pixel);
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


// #[test]
// fn test_opencv_findhomography() -> Result<(), Box<dyn Error>> {
//     use crate::filter;
//     use image::io::Reader as ImageReader;
//     use crate::harris::*;
//     use opencv::core::*;
//     use opencv::imgcodecs::*;
//     use opencv::imgproc::*;
//     use opencv::features2d::*;
//     use opencv::calib3d::*;
//     use opencv::types::*;

//     let img1_path = "hw/IMG_0627.JPG";
//     let img2_path = "hw/IMG_0628.JPG";
//     let img1 = ImageReader::open(img1_path)?.decode()?;
//     let img2 = ImageReader::open(img2_path)?.decode()?;
//     let gray1 = filter::filter(&img1, &filter::kernel::GaussianKernel::new(3, 2.0));
//     let gray2 = filter::filter(&img2, &filter::kernel::GaussianKernel::new(3, 2.0));

//     let corners1 = detect_harris_corners(&img1, HarrisCornerDetector::Harris, Some(0.04), 2.0, 2.0, 1.0, 500);
//     let pairs = generate_pairs(256, 9, &mut rand::thread_rng());
//     let descriptors = brief_descriptor(&gray1, &corners1, 9, &pairs);
//     let corners2 = detect_harris_corners(&img2, HarrisCornerDetector::Harris, Some(0.04), 2.0, 2.0, 1.0, 500);
//     let descriptors2 = brief_descriptor(&gray2, &corners2, 9, &pairs);
//     // opencv findhomograpy of two images
//     let matches = match_descriptors(&descriptors, &descriptors2, 0.1);
//     let src = matches.iter().map(|p| na::Point2::<f64>::new(p.first.x as f64, p.first.y as f64)).collect::<Vec<_>>();
//     let des = matches.iter().map(|p| na::Point2::<f64>::new(p.second.x as f64, p.second.y as f64)).collect::<Vec<_>>();
//     let src_mat = Mat::from_slice_2d(&src).unwrap();
//     let des_mat = Mat::from_slice_2d(&des).unwrap();
//     let mut h = Mat::default()?;
//     let mut mask = Mat::default()?;
//     find_homography(&src_mat, &des_mat, &mut h, RANSAC, 3.0, &mut mask, 1000, 0.995)?;
//     println!("{}", h);
//     assert!(1==2);
//     Ok(())
// }