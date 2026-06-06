pub type Float = f32;
pub type StateHandle = usize;

#[derive(Debug, Default)]
struct MarkovChain<State: Default> {
    states: Vec<State>,
    matrix: Vec<Vec<Float>>,
    vout_cache: Vec<Float>,
}

impl<State: Default> MarkovChain<State> {
    pub fn add_state(&mut self, state: State) -> StateHandle {
        let handle = self.states.len();
        self.states.push(state);
        let len = if let Some(v) = self.matrix.get(0) { v.len() } else { 0usize };
        self.matrix.push(vec![0.0; len]);
        for m in &mut self.matrix {
            m.push(0.0 as Float);
        }
        handle
    }

    pub fn set_edge(&mut self, state_a: usize, state_b: usize, value: Float) {
        self.matrix[state_b][state_a] = value;
    }

    pub fn get_edge(&mut self, state_a: usize, state_b: usize) -> Float {
        self.matrix[state_b][state_a]
    }

    pub fn new_output_vec(&mut self, init_state: usize) -> Vec<Float> {
        let mut out_vec = vec![0.0 as Float;self.matrix.len()];
        self.vout_cache = vec![0.0 as Float;self.matrix.len()];
        out_vec[init_state] = 1.0 as Float;
        out_vec
    }

    #[mathtrace::mathtrace]
    fn calc(&mut self, vout: &mut [f32]) {
        self.vout_cache.copy_from_slice(vout);
        for vi in 0..self.vout_cache.len() {
            let mut sum = 0.0 as Float;
            for ii in 0..self.matrix[vi].len() {
                sum += self.matrix[vi][ii] * self.vout_cache[ii];
            }
            vout[vi] = sum;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Based on the example graph shown in this video https://www.youtube.com/watch?v=1GKtfgwf3ig
    #[test]
    fn markovchain_works() {
        let mut chain = MarkovChain::<char>::default();

        let state_a = chain.add_state('A');
        let state_b = chain.add_state('B');

        chain.set_edge(state_a, state_a, 0.75);
        chain.set_edge(state_a, state_b, 0.25);
        chain.set_edge(state_b, state_b, 0.6);
        chain.set_edge(state_b, state_a, 0.4);

        let mut output = chain.new_output_vec(state_a);

        // Calculate first phase
        chain.calc(&mut output);
        println!("{:?}", output);

        // Calculate second phase
        chain.calc(&mut output);
        println!("{:?}", output);
    }
}
