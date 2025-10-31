use std::io::Cursor;
use std::io::Read;
use anyhow::Context;
use image::ImageReader;
use image::Rgba;
use neural::graph::*;
use neural::math::*;
use neural::sigmoid;

const TRAINING_ITER: usize = 1usize;
const LEARNING_RATE : f32 = 0.42; 

const CLASS: &[&'static str] = &[
    "apple",
    // "banana",
    "orange" 
];

fn get_expected(class: &'static str) -> [f32; CLASS.len()] {
    let mut arr = [0.0f32; CLASS.len()];
    let i = CLASS.iter().position(|x| x == &class).unwrap();
    arr[i] = 1.0;
    arr
}

fn activ_f(x: f32) -> f32 {
    sigmoid!(x)
}
fn main() -> anyhow::Result<()>{

    let image_size = 8usize*6usize;

    let mut graph = Graph::new(
        1.0, 
        image_size, 
        CLASS.len(), 
        5, 
        (&(activ_f as fn(Float) -> Float), 0.0, 1.0)
    );
    let mut cache = graph.new_cache();
    let mut output = vec![0.0f32; CLASS.len()];

    // Train for each classification
    for class in CLASS {
        let paths = std::fs::read_dir(&format!("./neural/examples/image-recognition/images/{}", class)).unwrap();
        for path in paths {
            if let Ok(r) = path {
                if !r.path().is_dir() {
                    println!("Training: {}", r.path().display());

                    let img_read = ImageReader::open(r.path())
                    .context("Could not open file!")?
                    .decode().context("Could not decode image!")?;

                    let img = match img_read.as_rgba8() {
                        Some(ii) => ii,
                        None => {
                            println!("Could not convert {:?} to RGBA", r.file_name());
                            continue
                        },
                    };
                    let input: Vec<f32> = img.pixels().map(|x|{
                        f32::from_be_bytes(x.0)
                    }).collect();
                    for _ in 0..TRAINING_ITER {
                        graph.train(&input, &mut cache, &mut output, &get_expected(class), LEARNING_RATE);
                    }
                }
            }
        }
    }

    // Run recognition tests with user input 
    // loop {
        // TODO: User input

        // TODO: Print recognization calculation 
    // }


    Ok(())
}