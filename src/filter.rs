
mod kernel {
use nalgebra as na;
// Define a trait for the kernel
pub trait Kernel {
    // Define a method to get the kernel size
    fn size(&self) -> usize;
    // Define a method to get the kernel values
    fn values(&self) -> &na::DMatrix<f64>;
}

pub struct GaussianKernel {
    values: na::DMatrix<f64>,
}

impl GaussianKernel {
    pub fn new(size: usize, sigma: f64) -> Self {
        let mut values = na::DMatrix::zeros(size, size);
        let mut sum = 0.0;
        for i in 0..size {
            for j in 0..size {
                let x = i as f64 - (size as f64 - 1.0) / 2.0;
                let y = j as f64 - (size as f64 - 1.0) / 2.0;
                values[(i, j)] = (-(x * x + y * y) / (2.0 * sigma * sigma)).exp();
                sum += values[(i, j)];
            }
        }
        let sum = values.sum();
        for i in 0..size {
            for j in 0..size {
                values[(i, j)] /= sum;
            }
        }
        Self { values }
    }
}

impl Kernel for GaussianKernel {
    fn size(&self) -> usize {
        self.values.nrows()
    }
    fn values(&self) -> &na::DMatrix<f64> {
        &self.values
    } 
}


}

mod test {
    use nalgebra::ComplexField;

    use super::kernel::*;
    #[test]
    fn test_gaussian_kernel() {
        let kernel = GaussianKernel::new(3, 2.0);
        println!("{}", kernel.values());
        assert_eq!(kernel.size(), 3);
        assert_eq!(kernel.values().nrows(), 3);
        assert_eq!(kernel.values().ncols(), 3);
        assert!((kernel.values().sum() - 1.0).abs() < 1e-6);
    }
}
