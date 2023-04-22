pub mod kernel {
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


use image::ImageBuffer;
use image::Luma;
use nalgebra as na;
use kernel::Kernel;

pub fn filter(input_image: &image::DynamicImage, kn: &dyn Kernel) -> image::GrayImage {
    let size = kn.size();
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
                    let kernel_value = kn.values()[(ky, kx)];

                    luma_sum += kernel_value * pixel[0] as f64;
                }
            }

            let new_pixel = Luma([luma_sum.round().min(255.0).max(0.0) as u8]);

            output_image.put_pixel(x, y, new_pixel);
        }
    }

    output_image
}

pub fn matrix_filter(input: &na::DMatrix<f64>, kn: &dyn Kernel) -> na::DMatrix<f64> {
    let size = kn.size();
    let (height, width) = input.shape();
    let mut output = na::DMatrix::zeros(height, width);

    for y in 0..height {
        for x in 0..width {
            let mut sum = 0f64;

            for ky in 0..size {
                for kx in 0..size {
                    let y_coord = (y as i32 + ky as i32 - size as i32).max(0).min(height as i32 - 1) as usize;
                    let x_coord = (x as i32 + kx as i32 - size as i32).max(0).min(width as i32 - 1) as usize;

                    let pixel = input[(y_coord, x_coord)];
                    let kernel_value = kn.values()[(ky, kx)];

                    sum += kernel_value * pixel as f64;
                }
            }

            output[(y, x)] = sum;
        }
    }

    output
}

mod test {
    use nalgebra as na;
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
        let kn = GaussianKernel::new(10, 2.0);
        let gaussian_filtered_img = super::filter(&img, &kn);
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

    #[test]
    fn test_sobel_filter() -> Result<(), Box<dyn Error>> {
        let img = ImageReader::open("hw/IMG_0627.JPG")?.decode()?;
        let kn_x = SobelKernel::new(Direction::X);
        let kn_y = SobelKernel::new(Direction::Y);
        let sobel_x_filtered_img = super::filter(&img, &kn_x);
        let sobel_y_filtered_img = super::filter(&img, &kn_y);
        sobel_x_filtered_img.save("soble_x.jpg")?;
        sobel_y_filtered_img.save("soble_y.jpg")?;

        Ok(())
    }

    #[test]
    fn test_matrix_filter() -> Result<(), Box<dyn Error>> {
        let luma2matrix = |img: &image::GrayImage| -> na::DMatrix<f64> {
            let (width, height) = img.dimensions();
            let mut matrix = na::DMatrix::zeros(height as usize, width as usize);
            for y in 0..height {
                for x in 0..width {
                    let pixel = img.get_pixel(x, y);
                    matrix[(y as usize, x as usize)] = pixel[0] as f64;
                }
            }
            matrix
        };
        let matrix2luma = |matrix: &na::DMatrix<f64>| -> image::GrayImage {
            let (height, width) = matrix.shape();
            let mut img = image::GrayImage::new(width as u32, height as u32);
            for y in 0..height {
                for x in 0..width {
                    let pixel = matrix[(y, x)] as u8;
                    img.put_pixel(x as u32, y as u32, image::Luma([pixel]));
                }
            }
            img
        };
        let img = ImageReader::open("hw/IMG_0627.JPG")?.decode()?;
        let input_image = img.to_luma8(); 
        let kn = GaussianKernel::new(10, 2.0);
        let input_matrix = luma2matrix(&input_image);
        let output_matrix = super::matrix_filter(&input_matrix, &kn);
        let output_image = matrix2luma(&output_matrix);
        output_image.save("matrix_filter.jpg")?;

        Ok(())
    }
}
