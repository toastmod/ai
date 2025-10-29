use std::io::Cursor;
use image::ImageReader;
use neural::graph::*;
use neural::math::*;
use neural::sigmoid;


const CLASS: &[&'static str] = &[
    "apple",
    "banana",
    "orange" 
];

fn activ_f(x: f32) -> f32 {
    sigmoid!(x)
}
fn main() -> anyhow::Result<()>{

    let image_size = 800usize*600usize;

    let mut graph = Graph::new(
        1.0, 
        image_size, 
        CLASS.len(), 
        5, 
        (&(activ_f as fn(Float) -> Float), 0.0, 1.0)
    );
    let mut cache = graph.new_cache();
    let mut output = vec![0.0f32; image_size];

    let paths = std::fs::read_dir("./examples/image-recognition/images/").unwrap();
    for path in paths {
        if let Ok(r) = path {
            if !r.path().is_dir() {
                println!("Training: {}", r.path().display());
                let img = ImageReader::open(r.path())?.decode()?;
                let input: Vec<f32> = img.as_bytes().iter().map(|x| x.clone() as f32).collect();
                // graph.train(&input, &mut cache, &output, expected_output, learning_rate);
            }
        }
    }

    Ok(())
}