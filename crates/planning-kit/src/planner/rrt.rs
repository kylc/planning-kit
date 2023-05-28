use petgraph::prelude::{NodeIndex, UnGraph};

use crate::{knn::nearest_neighbors_linear, state::StateSpace, validator::MotionValidator};

pub fn explore<S, Space, Validator>(
    space: &Space,
    validator: &Validator,
    steering_dist: f64,
    start: S,
    goal: S,
) -> UnGraph<S, f64>
where
    S: Clone,
    Space: StateSpace<State = S>,
    Validator: MotionValidator<State = S>,
{
    // Compute the requested number of states which are checked to reside in
    // the free space.
    let state_generator = std::iter::repeat_with(|| space.sample_uniform()).flatten();

    // Construct an undirected graph to be filled by the tree exploration.
    let mut graph = UnGraph::new_undirected();
    graph.add_node(start);

    for x_rand in state_generator {
        if let Some(&(x_nearest_ix, _)) =
            nearest_neighbors_linear(space, graph.node_weights(), &x_rand, 1).first()
        {
            let x_nearest = &graph[NodeIndex::new(x_nearest_ix)];
            let near_dist = space.distance(&x_rand, x_nearest);

            // The candidate point is formed by pulling from x_nearest toward
            // x_rand by the steering distance.
            let x_new = space.interpolate(
                x_nearest,
                &x_rand,
                (steering_dist / near_dist).clamp(0.0, 1.0),
            );

            let new_dist = space.distance(&x_new, x_nearest);
            let goal_dist = space.distance(&x_new, &goal);

            if validator.validate_motion(x_nearest, &x_new) {
                let x_new_ix = graph.add_node(x_new.clone());
                graph.add_edge(x_new_ix, NodeIndex::new(x_nearest_ix), new_dist);

                // Abort if a valid straight-line path is available directly to
                // the goal.
                if validator.validate_motion(&x_new, &goal) {
                    let goal_ix = graph.add_node(goal);
                    graph.add_edge(x_new_ix, goal_ix, goal_dist);

                    break;
                }
            }
        }
    }

    graph
}
