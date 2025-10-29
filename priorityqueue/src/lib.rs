use std::rc::Rc;

pub struct PriorityQueue<T: std::hash::Hash + Eq + Clone> {
    array: Vec<T>,
    // cmp_f: Box<dyn FnMut(&Rc<T>, &Rc<T>) -> bool>
}

impl<T: std::hash::Hash + Eq + Clone> PriorityQueue<T> {
    pub fn new(
        root: &T, 
        // cmp_f: Box<dyn FnMut(&Rc<T>, &Rc<T>) -> bool>
    ) -> Self {
        Self {
            array: vec![T::clone(root)],
            // cmp_f

        }
    }

    pub fn len(&self) -> usize {
        self.array.len()
    }

    pub fn enqueue(&mut self, state: &T, cmp_f: &mut dyn FnMut(&T, &T) -> bool) {

        let mut state = state;
        let mut new_idx = self.array.len();
        let mut parent_idx = f32::floor((new_idx-1) as f32 / 2.0f32) as usize;
        if state.eq(&self.array[parent_idx]) {
            return;
        }
        
        self.array.push(state.clone());
        loop {

            if cmp_f(&self.array[parent_idx], &self.array[new_idx]) {
                self.array.swap(new_idx, parent_idx);

                // when a swap occurs, move up the tree, select new state
                new_idx = parent_idx;   

                // Break when floated to the top
                if new_idx == 0 {
                    break;
                }

                parent_idx = f32::floor((new_idx-1) as f32 / 2.0f32) as usize; 

                // If we are no longer moving, then we've hit the top of the tree.
                if parent_idx == new_idx {
                    break;
                }
            } else {
                break;
            }

        }

    }

    pub fn dequeue_top(&mut self) -> Option<T> {
        if self.array.iter().len() >= 1 {
            Some(self.array.remove(0))
        } else {
            None
        }
    }

    pub fn dequeue_bottom(&mut self) -> Option<T> {
        self.array.pop()
    }
}


#[cfg(test)]
mod tests {

    use super::*;

    /// Based on [this example](https://www.geeksforgeeks.org/priority-queue-using-binary-heap/) from geeksforgeeks.org
    #[test]
    fn priorityqueue_works() {

        
        // Comparison function for the data format we are using
        let fb = &mut |a: &i32,b: &i32|{
            a < b
        };
        
        let mut queue = PriorityQueue::new(&45);
        queue.enqueue(&20, fb);
        queue.enqueue(&14, fb);
        queue.enqueue(&12, fb);
        queue.enqueue(&31, fb);
        queue.enqueue(&7, fb);
        queue.enqueue(&11, fb);
        queue.enqueue(&13, fb);
        queue.enqueue(&7, fb);

        let mut result = vec![];
        while let Some(val) = queue.dequeue_top() {
            result.push(val.clone());
        }

        assert_eq!(result.as_slice(), &[45, 31, 14, 13, 20, 7, 11, 12, 7])

    }
}
