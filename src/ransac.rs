use rand::{Rng, thread_rng};
use rand::seq::SliceRandom;

pub trait Model
where 
    Self::Point: Clone,
{
    type Point;
    type ModelParams;

    fn estimate_model(points: &[Self::Point]) -> Option<Self::ModelParams>;
    fn consensus_distance(params: &Self::ModelParams, point: &Self::Point) -> f64;
}

pub fn ransac<M: Model>(
    points: &[M::Point],
    n: usize,
    k: usize,
    t: f64,
    d: usize,
) -> Option<M::ModelParams> {
    let mut rng = thread_rng();
    let mut best_fit: Option<M::ModelParams> = None;
    let mut max_consensus = 0;

    for _ in 0..k {
        let sample_points: Vec<M::Point> = points.choose_multiple(&mut rng, n).cloned().collect();
        if let Some(model_params) = M::estimate_model(&sample_points) {
            let consensus_set: Vec<&M::Point> = points
                .iter()
                .filter(|&point| M::consensus_distance(&model_params, point) < t)
                .collect();

            if consensus_set.len() > max_consensus && consensus_set.len() >= d {
                max_consensus = consensus_set.len();
                best_fit = Some(model_params);
            }
        }
    }

    best_fit
}

#[cfg(test_ransac)]
mod test {
    use super::Model;
    use crate::harris::HarrisMatch;

    use nalgebra as na;

    impl Model for na::Matrix3<f64> {
        type Point = HarrisMatch;
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
    fn test_ransac_homo() -> Result<(), Box<dyn std::error::Error>> {
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

        println!("compute h naively");
        let des = matches.iter().map(|p| na::Point2::<f64>::new(p.first.x as f64, p.first.y as f64)).collect::<Vec<_>>();
        let src = matches.iter().map(|p| na::Point2::<f64>::new(p.second.x as f64, p.second.y as f64)).collect::<Vec<_>>();
        let h = crate::homography::compute_h(&des, &src)?;
        println!("{h}");

        let h = match super::ransac::<na::Matrix3<f64>>(&matches, 4, 10000, 3.0, matches.len() / 2) {
            Some(h) => h,
            None => return Err("no homography found".into()),
        };

        println!("{}", h);

        Ok(())
    }
}