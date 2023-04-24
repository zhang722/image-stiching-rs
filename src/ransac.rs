use rand::{Rng, thread_rng};
use rand::seq::SliceRandom;

pub trait Model {
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
        let sample_points: Vec<&M::Point> = points.choose_multiple(&mut rng, n).collect();
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

#[cfg(test)]
mod test {
    use super::Model;
    use crate::harris::HarrisMatch;

    use nalgebra as na;

    // struct Homography {

    // }

    impl Model for na::Matrix3<f64> {
        type Point = HarrisMatch;
        type ModelParams = na::Matrix3<f64>;

        fn estimate_model(points: &[Self::Point]) -> Option<Self::ModelParams> {
            
        }

        fn consensus_distance(params: &Self::ModelParams, point: &Self::Point) -> f64 {
            
        }
    }

    #[test]
    fn test_ransac_homo() -> Result<(), Box<dyn std::error::Error>> {
        

        Ok(())
    }
}