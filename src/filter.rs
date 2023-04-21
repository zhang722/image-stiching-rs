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
            for i in 0..size {
                for j in 0..size {
                    let x = i as f64 - (size as f64 - 1.0) / 2.0;
                    let y = j as f64 - (size as f64 - 1.0) / 2.0;
                    values[(i, j)] = (-(x * x + y * y) / (2.0 * sigma * sigma)).exp();
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

    pub enum Direction {
        X,
        Y,
    }
    pub struct SobelKernel {
        values: na::DMatrix<f64>,
    }

    impl SobelKernel {
        pub fn new(direction: Direction) -> Self {
            let values = match direction {
                Direction::X => 
                    na::DMatrix::from_row_slice(3, 3, &[
                        -1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0
                    ]),
                Direction::Y => 
                    na::DMatrix::from_row_slice(3, 3, &[
                        1.0, 2.0, 1.0, 0.0, 0.0, 0.0, -1.0, -2.0, -1.0
                    ])
            };
            Self { values }
        }
    }

    impl Kernel for SobelKernel {
        fn size(&self) -> usize {
            3
        }

        fn values(&self) -> &na::DMatrix<f64> {
            &self.values
        }
    }

}


mod filter {

    use image::ImageBuffer;
    use image::Luma;

    use super::kernel::Kernel;

    pub fn gaussian_filter(input_image: &image::DynamicImage, size: usize, sigma: f64) -> ImageBuffer<Luma<u8>, Vec<u8>> {
        let kernel = super::kernel::GaussianKernel::new(size, sigma);
        let input_image = input_image.to_luma8(); // 转换为灰度图像
        let (width, height) = input_image.dimensions();
        let mut output_image = ImageBuffer::new(width, height);

        for y in 0..height {
            for x in 0..width {
                let mut luma_sum = 0f64;

                for ky in 0..size {
                    for kx in 0..size {
                        let y_coord = (y as i32 + ky as i32 - size as i32).max(0).min(height as i32 - 1) as u32;
                        let x_coord = (x as i32 + kx as i32 - size as i32).max(0).min(width as i32 - 1) as u32;

                        let pixel = input_image.get_pixel(x_coord, y_coord);
                        let kernel_value = kernel.values()[(ky, kx)];

                        luma_sum += kernel_value * pixel[0] as f64;
                    }
                }

                let new_pixel = Luma([luma_sum.round().min(255.0).max(0.0) as u8]);

                output_image.put_pixel(x, y, new_pixel);
            }
        }

        output_image
    }

}

mod test {
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

    use std::error::Error;
    use image::io::Reader as ImageReader;
    #[test]
    fn test_gaussian_filter() -> Result<(), Box<dyn Error>> {
        let img = ImageReader::open("hw/IMG_0627.JPG")?.decode()?;
        let gaussian_filtered_img = super::filter::gaussian_filter(&img, 10, 2.0);
        gaussian_filtered_img.save("save.jpg")?;

        Ok(())
    }


    #[test]
    fn test_sobel_kernel() {
        let kernel_x = SobelKernel::new(Direction::X);
        let kernel_y = SobelKernel::new(Direction::Y);
        println!("{}", kernel_x.values());
        println!("{}", kernel_y.values());
        assert!(1==2);
    }
}
