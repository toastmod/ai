pub mod graph;

#[macro_use]
pub mod math;

#[cfg(test)]
mod tests {
    use crate::{graph::Graph, math::Float};

    use super::*;

    fn activation_function(x: Float) -> Float {
        x
    }

    #[test]
    fn it_works() {
        let mut graph = Graph::new(
            1.0, 
            5,
            8, 
            2, 
            10, 
            (
                &(activation_function as fn(Float) -> Float),
                1.0,
                1.0
            )
        );
        let mut cache = graph.new_cache(); 

        let input = vec![1.0; 10];
        let mut output = vec![0.0; 1];

        graph.calc_graph(&input, &mut cache, &mut output);

    }
}