use std::fs::DirEntry;
use std::io::Cursor;
use std::io::Read;
use anyhow::Context;
use image::ImageBuffer;
use image::ImageReader;
use image::Rgb;
use image::Rgba;
use image::imageops;
use image::math;
use neural::graph::*;
use neural::math::*;
use neural::sigmoid;
use show_image::{ create_window, ImageInfo, ImageView, WindowProxy };

// Number of training iterations. Present for future tuning.
const TRAINING_ITER: usize = 1usize;

// Learning rate used for training updates.
const LEARNING_RATE: Float = 0.42;

// Fixed image dimensions for this example.
const IMAGE_WIDTH: usize = 12;
const IMAGE_HEIGHT: usize = 12;
const IMAGE_SIZE: usize = IMAGE_WIDTH * IMAGE_HEIGHT;

// The set of classes used for the image recognition example.
const CLASS: &[&'static str] = &[
    "apple",
    "orange",
    // "dogs",
];

/// Return the class index for the given label.
fn get_index(class: &'static str) -> usize {
    CLASS.iter()
        .position(|x| x == &class)
        .unwrap()
}

/// Build the expected output vector for the provided class.
///
/// This example uses `1.0` for the target class and `-1.0` for all other classes.
fn get_expected(class: &'static str) -> [Float; CLASS.len()] {
    let mut arr = [1.0; CLASS.len()];
    let i = get_index(class);
    arr[i] = 2.0;
    arr
}

/// Convert an RGB pixel into a scalar input value.
///
/// This is a simple encoding used by the current neural graph.
fn rgb_to_float(rgb: [u8; 3]) -> Float {
    ((((rgb[0] as u32) << 16) + ((rgb[1] as u32) << 8) + (rgb[2] as u32)) as Float) /
        (16777215 as Float)

    // rgba[3] as Float * 4228255875.0
    // f32::from_ne_bytes(rgba) as Float
}

/// Update the preview window with the supplied image buffer.
///
/// Reuses the same window and image name so the content simply changes.
fn display_image(
    _window: &WindowProxy,
    _img: &ImageBuffer<Rgb<u8>, Vec<u8>>
) -> anyhow::Result<()> {
    // let image = ImageView::new(
    //     ImageInfo::rgb8(IMAGE_WIDTH as u32, IMAGE_HEIGHT as u32),
    //     img.as_raw()
    // );

    // window.set_image("preview", image)?;
    Ok(())
}

/// Load an image from disk, resize it to the fixed example dimensions,
/// and return the result as an RGB image buffer.
fn load_image(r: DirEntry) -> anyhow::Result<ImageBuffer<Rgb<u8>, Vec<u8>>> {
    let img_read = ImageReader::open(r.path())
        .context("Could not open file!")?
        .decode()
        .context("Could not decode image!")?
        .resize_exact(IMAGE_WIDTH as u32, IMAGE_HEIGHT as u32, imageops::FilterType::Nearest);

    let img: &ImageBuffer<Rgb<u8>, Vec<u8>> = img_read
        .as_rgb8()
        .context("Could not convert to RGB8")?;

    Ok(img.clone())
}

/// Activation function wrapper used by the neural graph.
// #[mathtrace::mathtrace]
fn activ_f(x: Float) -> Float {
    let xx = x / 60.0;
    1.0 / (1.0 + Float::powf(std::f32::consts::E as Float, -xx))
}

#[show_image::main]
fn main() -> anyhow::Result<()> {
    // Print some basic debug information and the target vectors.
    println!("===Debug===");
    for class in CLASS {
        println!("{} expected: {:?}", class, get_expected(class));
    }

    // Build the neural graph used for training and inference.
    let mut graph = Graph::new(1.0, IMAGE_SIZE, IMAGE_SIZE + 12, CLASS.len(), 5, (
        &(activ_f as fn(Float) -> Float),
        0.0,
    ));

    // graph.print();

    // Create a single preview window and reuse it for every image.
    let window = create_window("preview", Default::default())?;

    // Show one initial sample image before training begins.
    // if let Ok(mut paths) = std::fs::read_dir("./neural/examples/image-recognition/images/cats") {
    //     if
    //         let Some(Ok(first_entry)) = paths.find(|entry|
    //             entry
    //                 .as_ref()
    //                 .map(|e| !e.path().is_dir())
    //                 .unwrap_or(false)
    //         )
    //     {
    //         if let Ok(img) = load_image(first_entry) {
    //             let _ = display_image(&window, &img);
    //         }
    //     }
    // }

    let mut cache = graph.new_cache();
    let mut output: Vec<Float> = vec![0.0; CLASS.len()];

    // Train for each classification label.
    for class in CLASS {
        let paths = std::fs
            ::read_dir(&format!("./neural/examples/image-recognition/images/{}", class))
            .unwrap();

        let mut imgs_loaded = 0;
        'imgload: for path in paths {
            imgs_loaded += 1;
            if imgs_loaded >= 500 {
                break 'imgload;
            }
            if let Ok(r) = path {
                if !r.path().is_dir() {
                    let rr = r.path().display().to_string();

                    let img = match load_image(r) {
                        Ok(ii) => {
                            // println!("Training: {}", rr);
                            let _ = display_image(&window, &ii);
                            ii
                        }
                        Err(e) => {
                            println!("Error: {}", e);
                            continue;
                        }
                    };

                    // Convert the RGB pixels into the scalar input vector.
                    let input: Vec<Float> = img
                        .pixels()
                        .map(|x| rgb_to_float(x.0))
                        .collect();
                    // println!("{:?}", input);
                    // println!("----");

                    for _ in 0..TRAINING_ITER {
                        graph.train(
                            &input,
                            &mut cache,
                            &mut output,
                            &get_expected(class),
                            LEARNING_RATE
                        );
                        //      for o in 0..output.len() {
                        //          println!(
                        //              "{} | output: {:?} vs {:?}",
                        //              CLASS[o],
                        //              output[o],
                        //              get_expected(class)[o]
                        //          );
                        //      }
                    }
                }
            }
        }
    }

    // Reset training state before the test run.
    drop(cache);
    drop(output);
    let mut cache = graph.new_cache();
    let mut output = vec![0.0; CLASS.len()];

    // Test each classification label.
    for class in CLASS {
        let mut sum = vec![0.0; CLASS.len()];
        let mut count = 0.0;
        let paths = std::fs
            ::read_dir(&format!("./neural/examples/image-recognition/images/{}", class))
            .unwrap();

        for path in paths {
            count += 1.0;
            if let Ok(r) = path {
                if !r.path().is_dir() {
                    let img = match load_image(r) {
                        Ok(ii) => {
                            let _ = display_image(&window, &ii);
                            ii
                        }
                        Err(_) => {
                            continue;
                        }
                    };

                    let input: Vec<Float> = img
                        .pixels()
                        .map(|x| rgb_to_float(x.0))
                        .collect();
                    graph.calc_graph(&input, &mut cache, &mut output);

                    for o in 0..output.len() {
                        sum[o] += output[o];
                        // println!(
                        //     "{} | output: {} vs {}",
                        //     CLASS[o],
                        //     output[o],
                        //     get_expected(class)[o]
                        // );
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

    // Future interactive mode: allow the user to provide an image and run recognition live.
    // loop {
    // TODO: User input

    // TODO: Print recognition calculation
    // }

    Ok(())
}
