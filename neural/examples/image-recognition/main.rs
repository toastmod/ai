use std::fs::DirEntry;
use std::io::Cursor;
use std::io::Read;
use anyhow::Context;
use image::ImageBuffer;
use image::ImageReader;
use image::Rgba;
use image::imageops;
use neural::graph::*;
use neural::math::*;
use neural::sigmoid;

const TRAINING_ITER: usize = 1usize;
const LEARNING_RATE : Float = 0.42; 
const IMAGE_WIDTH: usize = 8;
const IMAGE_HEIGHT: usize = 8;
const IMAGE_SIZE: usize = IMAGE_WIDTH*IMAGE_HEIGHT; 

const CLASS: &[&'static str] = &[
    "apple",
    // "banana",
    "orange" 
];

fn get_index(class: &'static str) -> usize {
    CLASS.iter().position(|x| x == &class).unwrap()
}

fn get_expected(class: &'static str) -> [Float; CLASS.len()] {
    let mut arr = [0.0; CLASS.len()];
    let i = get_index(class); 
    arr[i] = 1.0;
    arr
}

fn rgba_to_float(rgba: [u8; 4]) -> Float {
    rgba[0] as Float * 255.0 + 
    rgba[1] as Float * 65025.0 + 
    rgba[2] as Float * 16581375.0 + 
    rgba[3] as Float * 4228255875.0
    // f32::from_ne_bytes(rgba) as Float
}

fn load_image(r: DirEntry) -> anyhow::Result<ImageBuffer<Rgba<u8>, Vec<u8>>> {
    let img_read = ImageReader::open(r.path())
    .context("Could not open file!")?
    .decode().context("Could not decode image!")?.resize_exact(IMAGE_WIDTH as u32, IMAGE_HEIGHT as u32, imageops::FilterType::Nearest);
    let img: &ImageBuffer<Rgba<u8>, Vec<u8>> = img_read.as_rgba8().context("Could not covert to RGBA8")?; 
    Ok(img.clone())
}

fn activ_f(x: Float) -> Float {
    sigmoid!(x)
}
fn main() -> anyhow::Result<()>{

    println!("===Debug===");
    for class in CLASS  {
        println!("{} expected: {:?}", class, get_expected(class))
    }

    let mut graph = Graph::new(
        2.0, 
        IMAGE_SIZE, 
        IMAGE_SIZE+4,
        CLASS.len(), 
        5, 
        (&(activ_f as fn(Float) -> Float), 1.0, 1.0)
    );

    graph.print();

    let mut cache = graph.new_cache();
    let mut output: Vec<Float> = vec![0.0; CLASS.len()];

    // Train for each classification
    for class in CLASS {
        let paths = std::fs::read_dir(&format!("./neural/examples/image-recognition/images/{}", class)).unwrap();
        for path in paths {
            if let Ok(r) = path {
                if !r.path().is_dir() {

                    // println!("Training: {}", r.path().display());
                    
                    let rr = r.path().display().to_string();
                    let img = match load_image(r) {
                        Ok(ii) => {
                            println!("Training: {}", rr);
                            println!("Sizematch: {}", (ii.len() == (IMAGE_SIZE*4)));
                            ii
                        },
                        Err(_) => continue,
                    };
                    
                    let input: Vec<Float> = img.pixels().map(|x|{
                        rgba_to_float(x.0) 
                    }).collect();


                    graph.train(&input, &mut cache, &mut output, &get_expected(class), LEARNING_RATE);
                    for o in 0..output.len() {
                        println!("{} | output: {} vs {}", CLASS[o], output[o], get_expected(class)[o]);
                    }
                    // for _ in 0..TRAINING_ITER {
                    // }
                }
            }
        }
    }

    drop(cache);
    drop(output);
    let mut cache = graph.new_cache(); 
    let mut output = vec![0.0; CLASS.len()];

    // Test for each classification
    for class in CLASS {
        let mut sum = vec![0.0; CLASS.len()];
        let mut count = 0.0;
        let paths = std::fs::read_dir(&format!("./neural/examples/image-recognition/images/{}", class)).unwrap();
        for path in paths {
            count += 1.0;
            if let Ok(r) = path {
                if !r.path().is_dir() {
                    let img = match load_image(r) {
                        Ok(ii) => ii,
                        Err(_) => continue,
                    };

                    let input: Vec<Float> = img.pixels().map(|x|{
                        rgba_to_float(x.0) 
                    }).collect();
                    graph.calc_graph(&input, &mut cache, &mut output);

                    for o in 0..output.len() {
                        sum[o] += output[o];
                        println!("{} | output: {} vs {}", CLASS[o], output[o], get_expected(class)[o]);
                    }

                }
            }
        }

        println!("Analysis of class: {}", class);
        // sum.iter().enumerate().for_each(|(idx, x)|{
        //     let avg = *x / count;
        //     println!("{} | avg: {}", CLASS[idx], avg);
        // });
        for idx in 0..sum.len() {
            let avg = sum[idx] / count;
            println!("{} | avg: {}", CLASS[idx], avg);
        }

    }

    // Run recognition tests with user input 
    // loop {
        // TODO: User input

        // TODO: Print recognization calculation 
    // }


    Ok(())
}