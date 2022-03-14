pub mod kdtree {
    use std::fmt;

    struct KDNode<T> {
        dim: usize,
        coords: Vec<f64>,
        subspace: Vec<f64>,
        axis: usize,
        is_right: bool,

        data: T,

        left: usize,
        right: usize
    }

    pub struct KDTree<T> {
        dim: usize,

        nodes: Vec<KDNode<T>>,
    }

    #[derive(Clone, Debug)]
    pub struct KDTreeError {
        pub msg: String,
    }

    impl fmt::Display for KDTreeError {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            write!(f, "{}", self.msg)
        }
    }

    impl<T> KDNode<T> {
        pub fn set_right(&mut self, id: usize, coords: &[f64], data: T) -> KDNode<T> {
            let mut node = KDNode {
                dim: self.dim,
                coords: coords.to_vec(),
                subspace: self.subspace.clone(),
                axis: self.next_axis(),
                is_right: true,
                left: 0,
                right: 0,
                data: data
            };
            node.subspace[self.axis*2] = self.coords[self.axis];
            self.right = id;
            node
        }

        pub fn set_left(&mut self, id: usize, coords: &[f64], data: T) -> KDNode<T> {
            let mut node = KDNode {
                dim: self.dim,
                coords: coords.to_vec(),
                subspace: self.subspace.clone(),
                axis: self.next_axis(),
                is_right: false,
                left: 0,
                right: 0,
                data: data
            };
            node.subspace[self.axis*2 + 1] = self.coords[self.axis];
            self.left = id;
            node
        }

        pub fn dist(&self, point: &[f64]) -> f64 {
            self.dist2(point).sqrt()
        }

        pub fn dist2(&self, point: &[f64]) -> f64 {
            let mut acc = 0.;
            for (i, c) in self.coords.iter().enumerate() {
                acc += (c - point[i])*(c - point[i]);
            }
            acc
        }

        #[allow(dead_code)]
        pub fn subspace_dist_to_point(&self, point: &[f64]) -> f64 {
            self.subspace_dist2_to_point(point).sqrt()
        }

        pub fn subspace_dist2_to_point(&self, point: &[f64]) -> f64 {
            let mut dist: Vec<f64> = vec![0.; self.dim];
            for (i, c) in point.iter().enumerate() {
                if *c > self.subspace[i*2] && *c < self.subspace[i*2 + 1] {
                    dist[i] = 0.;
                }
                else {
                    dist[i] = (self.subspace[i*2] - c).abs().min((self.subspace[i*2 + 1] - c).abs());
                }
            }
            dist.iter().fold(0., |acc, item| {
                acc + item*item
            })
        }

        pub fn next_axis(&self) -> usize {
            (self.axis+1)%self.dim
        }
    }

    impl<T> KDTree<T> {
        pub fn new(dim: usize) -> KDTree<T> {
            let tree: KDTree<T> = KDTree {
                dim: dim, nodes: Vec::new()
            };
            tree
        }

        pub fn size(&self) -> usize {
            self.nodes.len()
        }

        pub fn add_node(&mut self, coords: &[f64], data: T) -> usize {
            if self.nodes.is_empty() {
                let inf_subspace = vec![std::f64::NEG_INFINITY, std::f64::INFINITY, std::f64::NEG_INFINITY, std::f64::INFINITY, std::f64::NEG_INFINITY, std::f64::INFINITY];
                self.nodes.push(KDNode {
                dim: self.dim,
                coords: coords.to_vec(),
                subspace: inf_subspace,
                axis: 0,
                is_right: true,
                left: 0,
                right: 0,
                data: data
                });
                return 0
            }
            let new_id = self.nodes.len();
            let mut id_prev: usize;
            let mut id_next: usize = 0;
            loop {
                let axis = self.nodes[id_next].axis;
                id_prev = id_next;
                if self.nodes[id_next].coords[axis] > coords[axis] {
                    id_next = self.nodes[id_next].left;
                    if id_next == 0 {
                        let n = self.nodes[id_prev].set_left(new_id, coords, data);
                        self.nodes.push(n);
                        return new_id
                    }
                }
                else {
                    id_next = self.nodes[id_next].right;
                    if id_next == 0 {
                        let n = self.nodes[id_prev].set_right(new_id, coords, data);
                        self.nodes.push(n);
                        return new_id
                    }
                }
            }
        }

        pub fn nearest_neighbor(&self, coords: &[f64]) -> Result<usize, KDTreeError> {
            if self.nodes.is_empty() {
                return Err(KDTreeError {msg: "Empty tree.".to_owned()})
            }
            let mut path: Vec<usize> = Vec::new();
            let mut last_branch_right= true;
            let mut id_next: usize = 0;
            let mut root_node = true;
            while root_node || id_next > 0 {
                root_node = false;
                let axis = self.nodes[id_next].axis;
                path.push(id_next);
                if self.nodes[id_next].coords[axis] > coords[axis] {
                    id_next = self.nodes[id_next].left;
                    last_branch_right = false;
                }
                else {
                    id_next = self.nodes[id_next].right;
                    last_branch_right = true;
                }
            }

            let mut idx = path.len() - 1;
            let mut cur_best = path[idx];
            let mut min_dist = self.nodes[cur_best].dist2(coords);
            let mut at_root = false;
            while !at_root {
                let d = self.nodes[path[idx]].dist2(coords);
                if d < min_dist {
                    min_dist = d;
                    cur_best = path[idx];
                }
                if last_branch_right && self.nodes[path[idx]].left != 0 {
                    self.find_best(coords, self.nodes[path[idx]].left, &mut cur_best, &mut min_dist);
                }
                else if self.nodes[path[idx]].right != 0 {
                    self.find_best(coords, self.nodes[path[idx]].right, &mut cur_best, &mut min_dist);
                }
                last_branch_right = self.nodes[path[idx]].is_right;
                if idx == 0 {
                    at_root = true;
                }
                else {
                    idx -= 1;
                }
            }
            Ok(cur_best)
        }

        fn find_best(&self, coords: &[f64], node_id: usize, cur_best: &mut usize, min_dist: &mut f64) {
            if self.nodes[node_id].subspace_dist2_to_point(coords) > *min_dist {
                return;
            }
            let cur_dist = self.nodes[node_id].dist2(coords);
            if cur_dist < *min_dist {
                *min_dist = cur_dist;
                *cur_best = node_id;
            }
            if self.nodes[node_id].right > 0 && self.nodes[self.nodes[node_id].right].subspace_dist2_to_point(coords) < *min_dist {
                self.find_best(coords, self.nodes[node_id].right, cur_best, min_dist);
            }
            if self.nodes[node_id].left > 0 && self.nodes[self.nodes[node_id].left].subspace_dist2_to_point(coords) < *min_dist {
                self.find_best(coords, self.nodes[node_id].left, cur_best, min_dist);
            }
        }

        pub fn get_data(&self, id: usize) -> Result<&T, KDTreeError> {
            if id > self.nodes.len() - 1 {
                return Err(KDTreeError {msg: "Invalid id.".to_owned()})
            }
            Ok(&self.nodes[id].data)
        }

        pub fn get_coords(&self, id: usize) -> Result<Vec<f64>, KDTreeError> {
            if id > self.nodes.len() - 1 {
                return Err(KDTreeError {msg: "Invalid id.".to_owned()})
            }
            Ok(self.nodes[id].coords.clone())
        }
    
        pub fn get_dist(&self, id: usize, coords: &[f64]) -> Result<f64, KDTreeError> {
            if id > self.nodes.len() - 1 {
                return Err(KDTreeError {msg: "Invalid id.".to_owned()})
            }
            Ok(self.nodes[id].dist(coords))
        }
    }

    #[cfg(test)]
    mod tests {
        use super::KDTree;

        fn create_tree() -> KDTree<char> {
            let mut tree: KDTree<char> = KDTree {
                dim: 2, nodes: Vec::new()
            };
            let coords = vec![[7., 2.], [5., 4.], [9., 6.], [4., 7.], [8., 1.], [2., 3.]];
            let data = vec!['A', 'B', 'C', 'D', 'E', 'F'];

            let mut ids: Vec<usize> = Vec::new();

            for (i, c) in coords.iter().enumerate() {
                ids.push(tree.add_node(c, data[i]));
            }

            tree
        }

        #[test]
        fn test_tree_creation() {
            let mut tree = create_tree();

            assert_eq!(tree.size(), 6);
            
            tree.add_node(&[4., 5.], 'G');

            assert_eq!(tree.size(), 7);

            assert_eq!(tree.nodes[0].left, 1);
            assert_eq!(tree.nodes[0].right, 2);

            assert_eq!(tree.nodes[1].left, 5);
            assert_eq!(tree.nodes[1].right, 3);
        }

        #[test]
        fn test_nearest() {
            let tree = create_tree();

            assert_eq!(*tree.get_data(tree.nearest_neighbor(&[2., 5.]).unwrap()).unwrap(), 'F');
            assert_eq!(*tree.get_data(tree.nearest_neighbor(&[7., -2.]).unwrap()).unwrap(), 'E');
            assert_eq!(*tree.get_data(tree.nearest_neighbor(&[3., 3.]).unwrap()).unwrap(), 'F');
            assert_eq!(*tree.get_data(tree.nearest_neighbor(&[5., 3.]).unwrap()).unwrap(), 'B');
        }
    }
}
