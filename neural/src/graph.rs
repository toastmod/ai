use std::{collections::HashMap, vec};

use crate::{forw_calc, graph, math::*, sigmoid};

use tagmap::TagMap;

pub type EdgeHandle = usize;
pub type NodeHandle = usize;
pub type CacheHandle = usize;

pub struct Graph {
    nodes: TagMap<Node>,
    edges: TagMap<Edge>,
    input_layer: Vec<usize>,
    hidden_layers: Vec<Vec<usize>>,
    output_layer: Vec<usize>,
    cache: TagMap<(Float, bool)>,
    layer_size: usize,
}

impl Graph {
    pub fn new(init_weight: Float, layer_size: usize, output_layer_size: usize, hidden_layers_amt: usize, node_conf: (&'static fn(Float) -> Float, Float, Float) ) -> Self {
        let mut graph = Self {
            nodes: TagMap::new(), 
            edges: TagMap::new(), 
            input_layer: vec![], 
            hidden_layers: vec![], 
            output_layer: vec![], 
            cache: TagMap::new(), 
            layer_size
        };

        let (activ_f, activ_transform, activ_v) = node_conf;

        for _ in 0..graph.layer_size {
            graph.add_input_node(activ_f, activ_transform, activ_v);
        }

        for _ in 0..output_layer_size {
            graph.add_output_node(activ_f, activ_transform, activ_v);
        }

        for _ in 0..hidden_layers_amt {
            graph.add_layer(activ_f, activ_transform, activ_v);
        }

        graph.create_edges(init_weight);
        
        graph
    }

    fn add_node(&mut self, activ_f: &'static fn(Float) -> Float, activ_transform: Float, activ_v: Float) -> NodeHandle {
        let cache_handle = self.cache.add((0.0, false));
        self.nodes.add(Node::new(cache_handle, activ_f, activ_transform, activ_v))
    }

    fn add_input_node(&mut self, activ_f: &'static fn(Float) -> Float, activ_transform: Float, activ_v: Float) -> NodeHandle {
        let cache_handle = self.cache.add((0.0, false));
        let handle = self.nodes.add(Node::new(cache_handle, activ_f, activ_transform, activ_v));
        self.input_layer.push(handle);
        handle
    }

    fn add_output_node(&mut self, activ_f: &'static fn(Float) -> Float, activ_transform: Float, activ_v: Float) -> NodeHandle {
        let cache_handle = self.cache.add((0.0, false));
        let handle = self.nodes.add(Node::new(cache_handle, activ_f, activ_transform, activ_v));
        self.output_layer.push(handle);
        handle
    }

    fn add_layer(&mut self, activ_f: &'static fn(Float) -> Float, activ_transform: Float, activ_v: Float) {
        let mut layer = vec![];
        for _ in 0..self.layer_size {
            layer.push(self.add_node(activ_f, activ_transform, activ_v));
        }
        self.hidden_layers.push(layer);
    }

    /// Create edges with an init weight.
    fn create_edges(&mut self, init_weight: Float) {
        // Input layer
        for i in 0..self.input_layer.len() {
            for j in 0..self.hidden_layers[0].len() {
                let node_handle = self.input_layer[i];
                let hidden_node_handle = self.hidden_layers[0][i];
                self.connect(&node_handle, init_weight, &hidden_node_handle);
            }
        }

        // Hidden layers
        for window_i in (0..self.hidden_layers.len()).step_by(2) {
            // NOTE: A most unfortunate clone... thankfully this should only be called on init.
            let layer0 = self.hidden_layers[window_i].clone();
            let layer1 = self.hidden_layers[window_i+1].clone();
            for i in 0..layer0.len() {
                for j in 0..layer1.len() {
                    let node_handle = layer0[i];
                    let hidden_node_handle = layer1[i];
                    self.connect(&node_handle, init_weight, &hidden_node_handle);
                }
            }
        }

        // Output layer
        let final_hidden_i = self.hidden_layers.len()-1;
        for i in 0..self.output_layer.len() {
            for j in 0..self.hidden_layers[final_hidden_i].len() {
                let node_handle = self.input_layer[i];
                let hidden_node_handle = self.hidden_layers[final_hidden_i][i];
                self.connect(&node_handle, init_weight, &hidden_node_handle);
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
                an.forw.push(edge.clone())
            },
            None => todo!(),
        };

        match &mut self.nodes[b.clone()] {
            Some(bn) => {
                bn.back.push(edge);
            },
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

    pub fn calc_graph(&self, input: &[Float], cache: &mut TagMap<(Float, bool)>, output: &mut [Float]) {

        // Calculate input layer
        for i in 0..input.len() {
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

    pub fn train(&mut self, input: &[Float], cache: &mut TagMap<(f32, bool)>, output_buf: &mut Vec<Float>, expected_output: &[Float], learning_rate: Float) {
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
            let o_delta = o_y*(1.0-o_y)*(o_error);

            // Change output->hidden weight 
            for bi in node.back.clone() {
                let edge = self.edges[bi].as_ref().unwrap();
                let node1_cache = self.get_node(&edge.a).unwrap().cache_handle;
                let y1 = sigmoid!(cache[node1_cache].unwrap().0);
                let delta1 = y1*(1.0-y1)*(edge.weight * o_delta);
                error_cache[node1_cache].unwrap().0 = delta1;
                let edge_weight_delta = learning_rate*o_delta*y1;
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
                    let o_delta = y1*(1.0-y1)*(o_mul);
    
                    let delta1 = y1*(1.0-y1)*(edge.weight * o_delta);
                    let edge_weight_delta = learning_rate*o_delta*y1;
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
                let o_delta = y1*(1.0-y1)*(o_mul);
    
                let delta1 = y1*(1.0-y1)*(edge.weight * o_delta);
                let edge_weight_delta = learning_rate*o_delta*y1;
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
    activ_v: Float,
    cache_handle: CacheHandle,
}

impl Node {
    pub fn new(cache_handle: CacheHandle, activ_f: &'static fn(Float) -> Float, activ_transform: Float, activ_v: Float) -> Self {
        Self {
            forw: vec![],
            back: vec![],
            activ_f,
            activ_transform,
            activ_v,
            cache_handle,
        }
    }

    pub fn calc(&self, weighted_sum: Float) -> Float {
        forw_calc!(self.activ_f, weighted_sum, self.activ_transform,self.activ_v)
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
            b: b.clone()
        }
    }
}