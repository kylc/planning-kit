use petgraph::{
    algo::astar,
    prelude::{NodeIndex, UnGraph},
    visit::IntoNodeReferences,
};

use crate::{knn::nearest_neighbors_linear, state::StateSpace, validator::MotionValidator};

pub struct Roadmap<S> {
    pub graph: UnGraph<S, f64>,
}

impl<S: Clone> Roadmap<S> {
    /// Query the roadmap for the shortest path from the node nearest to the
    /// provided start state to the node nearest to the provided end state.
    ///
    /// If a path exists, returns a list of the visited states.
    pub fn query<Space>(&self, space: &Space, start: &S, goal: &S) -> Option<Vec<Space::State>>
    where
        Space: StateSpace<State = S>,
    {
        // TODO: Should probably add the start and goal states to the roadmap
        // and plan from them directly, rather than picking the closest already
        // existing states. As written, this will generally introduce collisions
        // in the first and final edge of the path.
        let start_ix = self.find_nearest_node(space, start)?;
        let goal_ix = self.find_nearest_node(space, goal)?;

        let (_, path) = astar(
            &self.graph,
            start_ix,
            |finish| finish == goal_ix,
            |e| *e.weight(),
            |ix| space.distance(&self.graph[ix], &self.graph[goal_ix]),
        )?;

        let mut path: Vec<_> = path.into_iter().map(|ix| self.graph[ix].clone()).collect();
        path.insert(0, start.clone());
        path.push(goal.clone());

        Some(path)
    }

    fn find_nearest_node<Space>(&self, space: &Space, state: &S) -> Option<NodeIndex>
    where
        Space: StateSpace<State = S>,
    {
        nearest_neighbors_linear(space, self.graph.node_weights(), state, 1)
            .first()
            .map(|&(idx, _)| NodeIndex::new(idx))
    }
}

pub fn build_roadmap<S, Space, Validator>(
    space: &Space,
    validator: &Validator,
    n: usize,
    k: usize,
) -> Roadmap<S>
where
    Space: StateSpace<State = S> + Clone,
    Validator: MotionValidator<State = S>,
{
    // Compute the requested number of states which are checked to reside in
    // the free space.
    let state_generator = std::iter::repeat_with(|| space.sample_uniform())
        .flatten()
        .filter(|s| validator.validate_state(s));

    // Construct an undirected graph of the free states.
    let mut graph = UnGraph::with_capacity(n, n * k); // TODO: better estimate edge count
    for state in state_generator.take(n) {
        graph.add_node(state);
    }

    // Connect the new node to all of its neighbors in the graph. This is the
    // most expensive step because it requires nearest neighbor searching of a
    // potentially large tree and discrete validation of intermediate connecting
    // paths, which may involve collision checking.
    //
    // Because of this, we make an effort to parallelize the operation.
    let node_refs = graph.node_references().collect::<Vec<_>>();
    let edges = node_refs
        .into_iter() // TODO: into_par_iter won't work for Python callbacks
        .flat_map(|(source_ix, source)| {
            // k + 1 neighbors because this will return the source itself as
            // neighbor 0.
            let neighbors = nearest_neighbors_linear(space, graph.node_weights(), source, k + 1);

            let mut edges = vec![];
            for (target_idx, dist) in neighbors.into_iter().skip(1) {
                let target = &graph[NodeIndex::new(target_idx)];
                if validator.validate_motion(source, target) {
                    edges.push((source_ix, NodeIndex::<u32>::new(target_idx), dist));
                }
            }
            edges
        })
        .collect::<Vec<_>>();

    // Coalesce the computed valid edges into the graph.
    for (source, target, dist) in edges {
        graph.add_edge(source, target, dist);
    }

    // The graph is full now and becomes immutable, so we might as well shrink
    // it to save some memory.
    graph.shrink_to_fit();

    Roadmap { graph }
}
