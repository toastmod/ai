use std::{ fs::File, io::Write, ops::IndexMut };

use crate::{ forw_calc, math::*, sigmoid, sigmoid_derivative };

use tagmap::TagMap;

pub type EdgeHandle = usize;
pub type NodeHandle = usize;
pub type CacheHandle = usize;

// macro_rules! dx {
//     () => {

//     };
// }

pub struct Graph {
    nodes: TagMap<Node>,
    edges: TagMap<Edge>,
    input_layer: Vec<usize>,
    hidden_layers: Vec<Vec<usize>>,
    output_layer: Vec<usize>,
    cache: TagMap<(Float, bool)>,
    hidden_layer_size: usize,
}

impl Graph {
    pub fn new(
        init_weight: Float,
        input_layer_size: usize,
        hidden_layer_size: usize,
        output_layer_size: usize,
        hidden_layers_amt: usize,
        node_conf: (&'static fn(Float) -> Float, Float)
    ) -> Self {
        let mut graph = Self {
            nodes: TagMap::new(),
            edges: TagMap::new(),
            input_layer: vec![],
            hidden_layers: vec![],
            output_layer: vec![],
            cache: TagMap::new(),
            hidden_layer_size,
        };

        let (activ_f, activ_transform) = node_conf;

        for _ in 0..input_layer_size {
            graph.add_input_node(activ_f, activ_transform);
        }

        for _ in 0..output_layer_size {
            graph.add_output_node(activ_f, activ_transform);
        }

        for _ in 0..hidden_layers_amt {
            graph.add_layer(activ_f, activ_transform);
        }

        graph.create_edges(init_weight);

        graph
    }

    fn vis_subgraph_layer(&self, cluster_name: &str, layer: &[usize]) -> String {
        let mut s = String::new();
        for n in layer {
            s.push_str(&format!("n{} ", n));
        }
        format!("subgraph cluster_{} {{ rank=same; {} }}", cluster_name, s)
    }

    fn vis_node_edges(&self, node: &Node) -> String {
        let mut s = String::from("{ ");
        for f in &node.forw {
            if let Some(fe) = &self.edges[f.clone()] {
                s.push_str(&format!("n{} ", fe.b));
            }
        }
        s.push('}');
        s
    }

    pub fn print(&self) {
        let mut input_sub = self.vis_subgraph_layer("input", &self.input_layer);
        input_sub.push('\n');
        let mut count = 0;
        for layer in &self.hidden_layers {
            input_sub.push_str(&self.vis_subgraph_layer(&format!("{}", count), layer));
            input_sub.push('\n');
            count += 1;
        }
        input_sub.push_str(&self.vis_subgraph_layer("output", &self.output_layer));
        input_sub.push('\n');

        for ni in 0..self.nodes.len() {
            if let Some(n) = &self.nodes[ni] {
                input_sub.push_str(&format!("n{} -> {};", ni, self.vis_node_edges(n)));
                input_sub.push('\n');
            }
        }

        let contents = format!("digraph {{\nrankdir=LR\n{}\n}}", input_sub);
        if let Ok(mut f) = File::create("graph.dot") {
            let _ = f.write_all(contents.as_bytes());
            println!("Wrote to file.")
            // graphviz_rust::exec(graph, ctx, args)
        } else {
            println!("ERROR!");
        }
    }

    fn add_node(
        &mut self,
        activ_f: &'static fn(Float) -> Float,
        activ_transform: Float
    ) -> NodeHandle {
        let cache_handle = self.cache.add((0.0, false));
        self.nodes.add(Node::new(cache_handle, activ_f, activ_transform))
    }

    fn add_input_node(
        &mut self,
        activ_f: &'static fn(Float) -> Float,
        activ_transform: Float
    ) -> NodeHandle {
        let cache_handle = self.cache.add((0.0, false));
        let handle = self.nodes.add(Node::new(cache_handle, activ_f, activ_transform));
        self.input_layer.push(handle);
        handle
    }

    fn add_output_node(
        &mut self,
        activ_f: &'static fn(Float) -> Float,
        activ_transform: Float
    ) -> NodeHandle {
        let cache_handle = self.cache.add((0.0, false));
        let handle = self.nodes.add(Node::new(cache_handle, activ_f, activ_transform));
        self.output_layer.push(handle);
        handle
    }

    fn add_layer(&mut self, activ_f: &'static fn(Float) -> Float, activ_transform: Float) {
        let mut layer = vec![];
        for _ in 0..self.hidden_layer_size {
            layer.push(self.add_node(activ_f, activ_transform));
        }
        self.hidden_layers.push(layer);
    }

    /// Create edges with an init weight.
    fn create_edges(&mut self, init_weight: Float) {
        // If there are no hidden layers, connect inputs directly to outputs.
        if self.hidden_layers.is_empty() {
            for input_handle in self.input_layer.clone() {
                for output_handle in self.output_layer.clone() {
                    self.connect(&input_handle, init_weight, &output_handle);
                }
            }
            return;
        }

        // Connect input layer to the first hidden layer.
        for i in 0..self.input_layer.len() {
            for j in 0..self.hidden_layer_size {
                let node_handle = self.input_layer[i];
                let hidden_node_handle = self.hidden_layers[0][j];
                self.connect(&node_handle, init_weight, &hidden_node_handle);
            }
        }

        // Connect hidden layers together.
        for window_i in 0..self.hidden_layers.len() - 1 {
            let layer0 = self.hidden_layers[window_i].clone();
            let layer1 = self.hidden_layers[window_i + 1].clone();
            for i in 0..layer0.len() {
                for j in 0..layer1.len() {
                    let node_handle = layer0[i];
                    let hidden_node_handle = layer1[j];
                    self.connect(&node_handle, init_weight, &hidden_node_handle);
                }
            }
        }

        // Connect the final hidden layer to the output layer.
        let final_hidden_i = self.hidden_layers.len() - 1;
        for i in 0..self.output_layer.len() {
            for j in 0..self.hidden_layers[final_hidden_i].len() {
                let hidden_node_handle = self.hidden_layers[final_hidden_i][j];
                let node_handle = self.output_layer[i];
                self.connect(&hidden_node_handle, init_weight, &node_handle);
            }
        }
    }

    pub fn new_cache(&self) -> TagMap<(Float, bool)> {
        let mut cache = self.cache.clone();
        for e in 0..self.cache.len() {
            cache[e].as_mut().unwrap().1 = false;
        }
        cache
    }

    fn connect(&mut self, a: &NodeHandle, weight: Float, b: &NodeHandle) {
        let edge = self.edges.add(Edge::new(a, weight, b));

        // Connect
        match &mut self.nodes[a.clone()] {
            Some(an) => {
                an.forw.push(edge.clone());
            }
            None => todo!(),
        }

        match &mut self.nodes[b.clone()] {
            Some(bn) => {
                bn.back.push(edge);
            }
            None => todo!(),
        };
    }

    pub fn get_node(&self, handle: &NodeHandle) -> Option<&Node> {
        self.nodes[handle.clone()].as_ref()
    }

    pub fn mut_node(&mut self, handle: &NodeHandle) -> Option<&mut Node> {
        self.nodes[handle.clone()].as_mut()
    }

    pub fn get_edge(&self, handle: &EdgeHandle) -> Option<&Edge> {
        self.edges[handle.clone()].as_ref()
    }

    pub fn mut_edge(&mut self, handle: &EdgeHandle) -> Option<&mut Edge> {
        self.edges[handle.clone()].as_mut()
    }

    pub fn calc_graph(
        &self,
        input: &[Float],
        cache: &mut TagMap<(Float, bool)>,
        output: &mut [Float]
    ) {
        // Calculate input layer
        for i in 0..self.input_layer.len() {
            let node = self.get_node(&self.input_layer[i]).unwrap();
            let cache_mem = cache[node.cache_handle].as_mut().unwrap();
            cache_mem.0 = node.calc(input[i]);
            cache_mem.1 = true;
        }

        // Calculate hidden layers
        for layer_i in 0..self.hidden_layers.len() {
            let layer = &self.hidden_layers[layer_i];
            for i in 0..layer.len() {
                // Get sum of incoming from cache
                let node = self.get_node(&layer[i]).unwrap();

                let mut sum = 0.0;
                for x in node.back.iter() {
                    let edge = self.get_edge(x).unwrap();
                    let cache_handle = self.get_node(&edge.a).as_ref().unwrap().cache_handle;

                    sum += cache[cache_handle].as_ref().unwrap().0 * edge.weight;

                    let cache_val = cache[cache_handle].as_ref().unwrap().0;
                }

                let cache_mem = cache[node.cache_handle].as_mut().unwrap();

                cache_mem.0 = node.calc(sum);
                cache_mem.1 = true;
            }
        }

        // Calculate output layers
        for i in 0..self.output_layer.len() {
            // Get sum of incoming from cache
            let node = self.get_node(&self.output_layer[i]).unwrap();

            // Get sum from result of previous nodes
            let mut sum = 0.0;
            for x in node.back.iter() {
                let edge = self.get_edge(x).unwrap();
                let cache_handle = self.get_node(&edge.a).as_ref().unwrap().cache_handle;

                sum += cache[cache_handle].as_ref().unwrap().0 * edge.weight;
            }

            let cache_mem = cache[node.cache_handle].as_mut().unwrap();

            let res = node.calc(sum);
            cache_mem.0 = res;
            cache_mem.1 = true;

            output[i] = res;
        }
    }

    pub fn train(
        &mut self,
        input: &[Float],
        cache: &mut TagMap<(Float, bool)>,
        output_buf: &mut Vec<Float>,
        expected_output: &[Float],
        learning_rate: Float
    ) {
        let mut error_cache = self.new_cache();

        // Note: Assumes a forward pass occured
        self.calc_graph(input, cache, output_buf);

        // Error check output nodes
        for i in 0..self.output_layer.len() {
            let expected = expected_output[i];
            let output = output_buf[i];
            let node = self.get_node(&self.output_layer[i]).unwrap();

            // Get output error
            let y_target = sigmoid!(expected);
            let o_y = sigmoid!(output);
            let o_error = y_target - o_y;
            let o_delta = o_y * (1.0 - o_y) * o_error;

            // Change output->hidden weight
            for bi in node.back.clone() {
                let edge = self.edges[bi].as_ref().unwrap();
                let node1_cache = self.get_node(&edge.a).unwrap().cache_handle;
                let y1 = sigmoid!(cache[node1_cache].unwrap().0);
                let delta1 = y1 * (1.0 - y1) * (edge.weight * o_delta);
                error_cache[node1_cache].unwrap().0 = delta1;
                let edge_weight_delta = learning_rate * o_delta * y1;
                let new_weight = edge_weight_delta + edge.weight;

                let edge = self.edges[bi].as_mut().unwrap();
                edge.weight = new_weight;
            }
        }

        // Change hidden layers (from back to front)
        for layer_i in (0..self.hidden_layers.len()).rev() {
            let hidden_layer = &self.hidden_layers[layer_i];

            for i in 0..self.hidden_layers[layer_i].len() {
                let node = self.get_node(&hidden_layer[i]).unwrap();
                let o_node_cache = node.cache_handle;

                // Change hidden->hidden weight
                for bi in node.back.clone() {
                    let edge = self.edges[bi].as_ref().unwrap();
                    let node1_cache = self.get_node(&edge.a).unwrap().cache_handle;
                    let y1 = sigmoid!(cache[node1_cache].unwrap().0);

                    let o_mul = edge.weight * error_cache[o_node_cache].unwrap().0;
                    let o_delta = y1 * (1.0 - y1) * o_mul;

                    let delta1 = y1 * (1.0 - y1) * (edge.weight * o_delta);
                    let edge_weight_delta = learning_rate * o_delta * y1;
                    let new_weight = edge_weight_delta + edge.weight;

                    let edge = self.edges[bi].as_mut().unwrap();
                    edge.weight = new_weight;
                }
            }
        }

        // Change hidden->input weight
        for i in 0..self.input_layer.len() {
            let node = self.get_node(&self.input_layer[i]).unwrap();
            let o_node_cache = node.cache_handle;

            // Change hidden->hidden weight
            for bi in node.back.clone() {
                let edge = self.edges[bi].as_ref().unwrap();
                let node1_cache = self.get_node(&edge.a).unwrap().cache_handle;
                let y1 = sigmoid!(cache[node1_cache].unwrap().0);

                let o_mul = edge.weight * error_cache[o_node_cache].unwrap().0;
                let o_delta = y1 * (1.0 - y1) * o_mul;

                let delta1 = y1 * (1.0 - y1) * (edge.weight * o_delta);
                let edge_weight_delta = learning_rate * o_delta * y1;
                let new_weight = edge_weight_delta + edge.weight;

                let edge = self.edges[bi].as_mut().unwrap();
                edge.weight = new_weight;
            }
        }
    }
}

pub struct Node {
    forw: Vec<EdgeHandle>,
    back: Vec<EdgeHandle>,
    activ_f: &'static fn(Float) -> Float,
    activ_transform: Float,
    cache_handle: CacheHandle,
}

impl Node {
    pub fn new(
        cache_handle: CacheHandle,
        activ_f: &'static fn(Float) -> Float,
        activ_transform: Float
    ) -> Self {
        Self {
            forw: vec![],
            back: vec![],
            activ_f,
            activ_transform,
            cache_handle,
        }
    }

    pub fn calc(&self, weighted_sum: Float) -> Float {
        let r = forw_calc!(self.activ_f, weighted_sum, self.activ_transform);
        // println!("f({:?}) => {:?}", weighted_sum, r);
        r
    }
}

pub struct Edge {
    weight: Float,
    a: NodeHandle,
    b: NodeHandle,
}

impl Edge {
    pub fn new(a: &NodeHandle, weight: Float, b: &NodeHandle) -> Self {
        Self {
            weight,
            a: a.clone(),
            b: b.clone(),
        }
    }
}
