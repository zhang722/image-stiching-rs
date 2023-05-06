mod harris;
mod filter;
mod homography;
mod ransac;
mod stitching;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let input_dir = std::path::Path::new("data");
    let output_dir = std::path::Path::new("gaussian");
    filter::noise::process_images(input_dir, output_dir, &filter::noise::NoiseType::Gaussian).unwrap();
    let output_dir = std::path::Path::new("salt_and_pepper");
    filter::noise::process_images(input_dir, output_dir, &filter::noise::NoiseType::SaltAndPepper).unwrap();

    let mut subdirectories = Vec::new();

    for entry in std::fs::read_dir("data")? {
        let entry = entry?;
        let path = entry.path();

        if path.is_dir() {
            if let Some(dir_name) = path.file_name() {
                subdirectories.push(dir_name.to_string_lossy().into_owned());
            }
        }
    }

    let img_paths = ["data", "gaussian", "salt_and_pepper"];
    for img_path in img_paths.iter().take(1) {
        for subdirectory in subdirectories.iter() {
            let input_dir = std::path::Path::new(img_path).join(subdirectory);
            match stitching::stitching_dir(input_dir, 2000, 3.0) {
                Ok(_) => {},
                Err(e) => println!("stitching dir failed:{}", e),  
            }
        }
    }

    for img_path in img_paths.iter().skip(1) {
        for subdirectory in subdirectories.iter() {
            let input_dir = std::path::Path::new(img_path).join(subdirectory);
            match stitching::stitching_dir(input_dir, 3000, 5.0) {
                Ok(_) => {},
                Err(e) => println!("stitching dir failed:{}", e),  
            }
        }
    }

    Ok(())
}
