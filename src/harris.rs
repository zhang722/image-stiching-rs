use std::error::Error;
use image::{imageops, ImageBuffer, Rgb, Rgba};
use image::{io::Reader as ImageReader, GenericImage, GenericImageView};
use imageproc::drawing::BresenhamLineIter;
use nalgebra as na;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use rand::rngs::ThreadRng;

fn luma2matrix(img: &image::GrayImage) -> na::DMatrix<f64> {
    let (width, height) = img.dimensions();
    let mut matrix = na::DMatrix::zeros(height as usize, width as usize);
    for y in 0..height {
        for x in 0..width {
            let pixel = img.get_pixel(x, y);
            matrix[(y as usize, x as usize)] = pixel[0] as f64;
        }
    }
    matrix
}

fn matrix2luma(matrix: &na::DMatrix<f64>) -> image::GrayImage {
    let (height, width) = matrix.shape();
    let mut img = image::GrayImage::new(width as u32, height as u32);
    for y in 0..height {
        for x in 0..width {
            let pixel = matrix[(y, x)] as u8;
            img.put_pixel(x as u32, y as u32, image::Luma([pixel]));
        }
    }
    img
}

pub fn compute_autocorrelation_matrix(gradient_x: image::GrayImage, gradient_y: image::GrayImage, sigma: f64) 
-> (na::DMatrix<f64>, na::DMatrix<f64>, na::DMatrix<f64>) {
    let gradient_x_matrix = luma2matrix(&gradient_x);
    let gradient_y_matrix = luma2matrix(&gradient_y);
    let (nrows, ncols) = gradient_x_matrix.shape();
    let ixix = na::DMatrix::<f64>::from_vec(
        nrows, 
        ncols, 
        gradient_x_matrix.iter().map(|f| f.powi(2)).collect::<Vec<f64>>()
    );
    let ixiy = na::DMatrix::<f64>::from_vec(
        nrows, 
        ncols, 
        gradient_x_matrix.iter().zip(gradient_y_matrix.iter()).map(|(ix, iy)| ix * iy).collect::<Vec<f64>>()
    );
    let iyiy = na::DMatrix::<f64>::from_vec(
        nrows, 
        ncols, 
        gradient_y_matrix.iter().map(|f| f.powi(2)).collect::<Vec<f64>>()
    );
    let kn = crate::filter::kernel::GaussianKernel::new(3, sigma);

    (crate::filter::matrix_filter(&ixix, &kn),
        crate::filter::matrix_filter(&ixiy, &kn),
        crate::filter::matrix_filter(&iyiy, &kn))
}

pub enum HarrisCornerDetector {
    Harris,
    ShiTomasi,
    HarmonicMean,
}
    
fn compute_corner_response(
    ixix: &na::DMatrix<f64>, 
    ixiy: &na::DMatrix<f64>, 
    iyiy: &na::DMatrix<f64>,
    measure: HarrisCornerDetector,
    k: Option<f64>
) -> na::DMatrix<f64> {
    let (nrows, ncols) = ixix.shape();
    let mut response = na::DMatrix::<f64>::zeros(nrows, ncols);

    // |a b| = |ixix ixiy|
    // |c a|   |ixiy iyiy|
    let compute_response_with_harris_measure = |a: f64, b: f64, c: f64, k: Option<f64>| -> f64 {
        let det = a * c - b.powi(2);
        let trace = a + c;
        det - k.expect("Using Harris Measure, but no k given!") * trace.powi(2)
    };
    let compute_response_with_shi_tomasi_measure = |a: f64, b: f64, c: f64| -> f64 {
        (a + c - ((a - c).powi(2) + 4.0 * b.powi(2)).sqrt()) * 0.5
    };
    let compute_response_with_harmonic_mean_measure = |a: f64, b: f64, c: f64| -> f64 {
        let det = a * c - b.powi(2);
        let trace = a + c;
        det / trace
    };
    for y in 0..nrows {
        for x in 0..ncols {
            match measure {
                HarrisCornerDetector::Harris => {
                    response[(y, x)] = compute_response_with_harris_measure(
                        ixix[(y, x)], ixiy[(y, x)], iyiy[(y, x)], k
                    );
                },
                HarrisCornerDetector::ShiTomasi => {
                    response[(y, x)] = compute_response_with_shi_tomasi_measure(
                        ixix[(y, x)], ixiy[(y, x)], iyiy[(y, x)]
                    );
                },
                HarrisCornerDetector::HarmonicMean => {
                    response[(y, x)] = compute_response_with_harmonic_mean_measure(
                        ixix[(y, x)], ixiy[(y, x)], iyiy[(y, x)]
                    );
                },
            }
        }
    }
    response
}

fn non_maximum_suppression(corner_response: na::DMatrix<f64>, tau: f64,  window_size: usize) -> na::DMatrix<f64> {
    let (height, width) = corner_response.shape();
    let mut suppressed_corner_response = na::DMatrix::zeros(height, width);

    for y in 0..height {
        for x in 0..width {
            if corner_response[(y, x)] < tau {
                continue;
            }

            let mut max_response = corner_response[(y, x)];
            let half_window = window_size / 2;

            for ky in std::cmp::max(0, y as isize - half_window as isize) as usize
                ..std::cmp::min(height, y + half_window + 1)
            {
                for kx in std::cmp::max(0, x as isize - half_window as isize) as usize
                    ..std::cmp::min(width, x + half_window + 1)
                {
                    max_response = max_response.max(corner_response[(ky, kx)]);
                }
            }

            suppressed_corner_response[(y, x)] = if corner_response[(y, x)] == max_response {
                corner_response[(y, x)]
            } else {
                0.0
            };
        }
    }

    suppressed_corner_response
}

#[derive(Debug, Clone, Copy)]
pub struct HarrisCorner {
    pub x: usize,
    pub y: usize,
    pub response: f64,
}

pub fn select_output_corners(
    corner_response: na::DMatrix<f64>,
    // cells: (usize, usize),
    num: usize
) -> Vec<HarrisCorner> {
    let mut corners = Vec::<HarrisCorner>::new();
    let (height, width) = corner_response.shape();

    for y in 0..height {
        for x in 0..width {
            if corner_response[(y, x)] > 0.0 {
                corners.push(HarrisCorner {
                    x,
                    y,
                    response: corner_response[(y, x)],
                });
            }
        }
    }

    corners.sort_by(|a, b| b.response.partial_cmp(&a.response).unwrap());
    corners.truncate(num);

    corners
}

pub fn draw_corners(img: &image::DynamicImage, corners: &Vec<HarrisCorner>) -> image::DynamicImage {
    let mut img = img.clone();
    let (width, height) = img.dimensions();
    for corner in corners {
        let x = corner.x;
        let y = corner.y;
        let r = 5;
        for i in 0..r {
            for j in 0..r {
                // if (i as f64 - r as f64 / 2.0).powi(2) + (j as f64 - r as f64 / 2.0).powi(2) < (r as f64 / 2.0).powi(2) {
                if i == 0 || i + 1 == r || j == 0 || j + 1 == r {
                    if (x as u32 + i as u32) < width && (y as u32 + j as u32) < height {
                        img.put_pixel(x as u32 + i as u32, y as u32 + j as u32, image::Rgba([255, 0, 0, 255]));
                    }
                }
            }
        }
    }
    img
}

#[derive(Debug)]
pub struct Brief {
    pub descriptor: [u32; 8],
    pub harris_corner: HarrisCorner,
}

#[derive(Clone)]
pub struct HarrisMatch {
    pub first: HarrisCorner,
    pub second: HarrisCorner,
    pub distance: u32,
}

impl HarrisMatch {
    fn new(first: HarrisCorner, second: HarrisCorner, distance: u32) -> Self {
        Self {
            first,
            second,
            distance,
        }
    }
}

pub fn generate_pairs(descriptor_size: usize, patch_size: usize, rng: &mut ThreadRng) -> Vec<((u32, u32), (u32, u32))> {
    (0..descriptor_size)
        .map(|_| {
            (
                (rng.gen_range(0..patch_size) as u32, rng.gen_range(0..patch_size) as u32),
                (rng.gen_range(0..patch_size) as u32, rng.gen_range(0..patch_size) as u32),
            )
        })
        .collect()
}

pub fn brief_descriptor(image: &image::GrayImage, keypoints: &[HarrisCorner], patch_size: usize, pairs: &[((u32, u32), (u32, u32))]) -> Vec<Brief> {
    let mut descriptors = Vec::with_capacity(keypoints.len());

    for &HarrisCorner{x, y, response } in keypoints {
        let x = x as u32;
        let y = y as u32;
        let mut bitstring: [u32; 8] = [0; 8];

        for (i, &((x1, y1), (x2, y2))) in pairs.iter().enumerate() {
            let p1 = x1.wrapping_add(x) as i32 - (patch_size / 2) as i32;
            let q1 = y1.wrapping_add(y) as i32 - (patch_size / 2) as i32;
            let p2 = x2.wrapping_add(x) as i32 - (patch_size / 2) as i32;
            let q2 = y2.wrapping_add(y) as i32 - (patch_size / 2) as i32;

            if p1 >= 0 && q1 >= 0 && p2 >= 0 && q2 >= 0 && p1 < image.width() as i32 && q1 < image.height() as i32 && p2 < image.width() as i32 && q2 < image.height() as i32 {
                let intensity1 = image.get_pixel(p1 as u32, q1 as u32).0[0] as i32;
                let intensity2 = image.get_pixel(p2 as u32, q2 as u32).0[0] as i32;

                if intensity1 < intensity2 {
                    bitstring[i / 32] |= 1 << (i % 32);
                }
            }
        }

        descriptors.push(Brief {
            descriptor: bitstring,
            harris_corner: HarrisCorner {
                x: x as usize,
                y: y as usize,
                response: response,
            },
        });
    }

    descriptors
}

fn hamming_distance(a: &Brief, b: &Brief) -> u32 {
    let mut distance = 0;
    let a = a.descriptor;
    let b = b.descriptor;
    for (x, y) in a.iter().zip(b.iter()) {
        distance += (x ^ y).count_ones();
    }
    distance
}

pub fn match_descriptors(descriptors1: &[Brief], descriptors2: &[Brief], threshold: f64) -> Vec<HarrisMatch> {
    let mut matches = Vec::new();
    let threshold = (threshold * 256.0) as u32;

    for descriptor1 in descriptors1 {
        let mut best_match = None;
        let mut best_distance = std::u32::MAX;

        for descriptor2 in descriptors2 {
            let distance = hamming_distance(descriptor1, descriptor2);
            if distance < best_distance {
                best_distance = distance;
                best_match = Some(descriptor2);
            }
        }

        if let Some(best_match) = best_match {
            if best_distance > threshold {
                continue;
            }
            matches.push(HarrisMatch::new(
                descriptor1.harris_corner,
                best_match.harris_corner,
                best_distance,
            ));
        }
    }

    matches
}

pub fn draw_matches(img1: &image::DynamicImage, img2: &image::DynamicImage, matches: &[HarrisMatch]) -> image::DynamicImage {
    let (width1, height1) = img1.dimensions();
    let (width2, height2) = img2.dimensions();

    let total_width = width1 + width2;
    let max_height = height1.max(height2);

    let mut combined_image = image::DynamicImage::new_rgb8(total_width, max_height);

    // Copy img1 and img2 into combined_image
    imageops::replace(&mut combined_image, img1, 0, 0);
    imageops::replace(&mut combined_image, img2, width1.into(), 0);

    for pair in matches {

        let (x1, y1) = (pair.first.x as f32, pair.first.y as f32);
        let (x2, y2) = (pair.second.x as f32, pair.second.y as f32);
        let x2_offset = x2 + width1 as f32;
        imageproc::drawing::draw_line_segment_mut(&mut combined_image, (x1, y1), (x2_offset, y2), Rgba([0, 255, 0, 255]));
    }
    
    combined_image
}

pub fn detect_harris_corners(
    img: &image::DynamicImage, 
    measure: HarrisCornerDetector,
    k: Option<f64>,
    sigma_d: f64,
    sigma_i: f64,
    tau: f64,
    num: usize,
) -> Vec<HarrisCorner> {
    use crate::filter;
    let gray = filter::filter(&img, &filter::kernel::GaussianKernel::new(3, sigma_d));
    let gray = image::DynamicImage::ImageLuma8(gray);
    let gx = filter::filter(&gray, &filter::kernel::SobelKernel::new(filter::kernel::Direction::X));
    let gy = filter::filter(&gray, &filter::kernel::SobelKernel::new(filter::kernel::Direction::Y));
    let (a, b, c) = compute_autocorrelation_matrix(gx, gy, sigma_i);
    let res = compute_corner_response(&a, &b, &c, measure, k);
    let res = non_maximum_suppression(res, tau, 10);
    select_output_corners(res, num)
}

#[test]
fn test_compute_response() -> Result<(), Box<dyn std::error::Error>> {
    use crate::filter;

    let img = ImageReader::open("hw/IMG_0627.JPG")?.decode()?;
    let gx = filter::filter(&img, &filter::kernel::SobelKernel::new(filter::kernel::Direction::X));
    let gy = filter::filter(&img, &filter::kernel::SobelKernel::new(filter::kernel::Direction::Y));
    let gray = filter::filter(&img, &filter::kernel::GaussianKernel::new(3, 2.0));

    let (a, 
        b, 
        c) = compute_autocorrelation_matrix(gx, gy, 2.0);
    let res = compute_corner_response(&a, &b, &c, HarrisCornerDetector::Harris, Some(0.04));
    let res = non_maximum_suppression(res, 1.0, 10);
    let corners = select_output_corners(res, 10);
    let descriptors = brief_descriptor(&gray, &corners, 9, &generate_pairs(256, 9, &mut rand::thread_rng()));
    let d1 = &descriptors[0];
    for d in d1.descriptor {
        println!("{:#b}", d);
    }

    // img.save("res_harris.png")?;
    assert!(1==2);


    Ok(())
}


#[test]
fn show_pair_images() -> Result<(), Box<dyn std::error::Error>> {
    let img1_path = "hw/IMG_0627.JPG";
    let img2_path = "hw/IMG_0628.JPG";
    let img1 = image::open(img1_path).expect("Failed to open image 1").to_rgb8();
    let img2 = image::open(img2_path).expect("Failed to open image 2").to_rgb8();

    let (width1, height1) = img1.dimensions();
    let (width2, height2) = img2.dimensions();

    let total_width = width1 + width2;
    let max_height = height1.max(height2);

    let mut combined_image = ImageBuffer::<Rgb<u8>, _>::new(total_width, max_height);

    // Copy img1 and img2 into combined_image
    imageops::replace(&mut combined_image, &img1, 0, 0);
    imageops::replace(&mut combined_image, &img2, width1.into(), 0);

    // Corresponding points (x1, y1) in img1 and (x2, y2) in img2
    let points = vec![
        ((50, 60), (40, 70)),
        ((100, 120), (90, 130)),
    ];

    for &((x1, y1), (x2, y2)) in &points {
        let x2_offset = x2 + width1;
        imageproc::drawing::draw_line_segment_mut(&mut combined_image, (x1 as f32, y1 as f32), (x2_offset as f32, y2 as f32), Rgb([0, 255, 0]));
    }

    combined_image.save("output.jpg").expect("Failed to save output image");

    Ok(())
}

#[test]
fn test_draw_matches() -> Result<(), Box<dyn std::error::Error>> {
    use crate::filter;

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
    for m in &matches {
        println!("{}", m.distance,);
    }

    let output = draw_matches(&img1, &img2, &matches);
    output.save("output.jpg").expect("Failed to save output image");
    assert!(1==2);
    Ok(())
}