use std::error::Error;
use image::io::Reader as ImageReader;

pub fn read_img(path: &str) -> Result<image::DynamicImage, Box<dyn Error>> {
    let img = ImageReader::open(path)?.decode()?;
    img.save("save.jpg")?;
    Ok(img)
}

#[test] 
fn test_read_img() -> Result<(), Box<dyn Error>> {
    read_img("hw/IMG_0627.JPG")?;
    Ok(())
}