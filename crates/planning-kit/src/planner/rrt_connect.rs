use std::{collections::HashMap, mem};

use petgraph::{
    prelude::{NodeIndex, UnGraph},
    visit::{EdgeRef, IntoNodeReferences},
};

use crate::{knn::nearest_neighbors_linear, state::StateSpace, validator::MotionValidator};

#[derive(Copy, Clone, Eq, PartialEq)]
enum ExtensionResult {
    Trapped,
    Reached(NodeIndex),
    Advanced(NodeIndex),
}

fn extend<S, Space, Validator>(
    graph: &mut UnGraph<S, f64>,
    space: &Space,
    validator: &Validator,
    q: &S,
    steering_dist: f64,
) -> ExtensionResult
where
    S: Clone + PartialEq,
    Space: StateSpace<State = S>,
    Validator: MotionValidator<State = S>,
{
    // SAFETY: The provided graph is assumed to have at least one node (either
    // the start or the goal), so we can safely unwrap it.
    let &(q_near_ix, q_near_dist) = nearest_neighbors_linear(space, graph.node_weights(), q, 1)
        .first()
        .unwrap();

    let q_near = &graph[NodeIndex::new(q_near_ix)];

    // If we are within the steering distance of `q` then there is no sense
    // in interpolating toward it--just add it to the tree directly.
    // Otherwise, steer toward `q` as in vanilla RRT.
    let q_new = if space.distance(q_near, q) < steering_dist {
        q.clone()
    } else {
        space.interpolate(q_near, q, (steering_dist / q_near_dist).clamp(0.0, 1.0))
    };

    if validator.validate_motion(q_near, &q_new) {
        let q_new_dist = space.distance(q_near, &q_new);

        let q_new_ix = graph.add_node(q_new.clone());
        graph.add_edge(NodeIndex::new(q_near_ix), q_new_ix, q_new_dist);

        if &q_new == q {
            ExtensionResult::Reached(q_new_ix)
        } else {
            ExtensionResult::Advanced(q_new_ix)
        }
    } else {
        ExtensionResult::Trapped
    }
}

fn connect<S, Space, Validator>(
    graph: &mut UnGraph<S, f64>,
    space: &Space,
    validator: &Validator,
    q: &S,
    steering_dist: f64,
) -> ExtensionResult
where
    S: Clone + PartialEq,
    Space: StateSpace<State = S>,
    Validator: MotionValidator<State = S>,
{
    // Continue advancing the tree as long as we don't reach q nor get trapped.
    loop {
        match extend(graph, space, validator, q, steering_dist) {
            ExtensionResult::Advanced(_) => {}
            x => return x,
        }
    }
}

pub fn explore<S, Space, Validator>(
    space: &Space,
    validator: &Validator,
    steering_dist: f64,
    start: S,
    goal: S,
    n: usize,
) -> UnGraph<S, f64>
where
    S: Clone + PartialEq,
    Space: StateSpace<State = S>,
    Validator: MotionValidator<State = S>,
{
    // Compute the requested number of states which are checked to reside in
    // the free space.
    let state_generator = std::iter::repeat_with(|| space.sample_uniform()).flatten();

    // Construct an undirected graph to be filled by the tree exploration.
    let mut t_a = UnGraph::new_undirected();
    t_a.add_node(start);

    let mut t_b = UnGraph::new_undirected();
    t_b.add_node(goal);

    for q_rand in state_generator.take(n) {
        match extend(&mut t_a, space, validator, &q_rand, steering_dist) {
            // In this case, the extension of T_a has reached or at least
            // advanced toward our q_rand state.
            ExtensionResult::Reached(q_a_new_ix) | ExtensionResult::Advanced(q_a_new_ix) => {
                let q_new = &t_a[q_a_new_ix];

                // So we try to connect to this new advancement of T_a by growing T_b.
                if let ExtensionResult::Reached(q_b_new_ix) =
                    connect(&mut t_b, space, validator, q_new, steering_dist)
                {
                    // T_b has reached a node on T_a. This point q_new is our
                    // keystone for connecting a path through T_a and T_b
                    // between the start and goal.
                    let fulcrum_dist = space.distance(&t_a[q_a_new_ix], &t_b[q_b_new_ix]);
                    return merge_graphs(t_a, t_b, Some((q_a_new_ix, q_b_new_ix, fulcrum_dist)));
                }
            }

            // Extending T_a got trapped. Don't bother trying to attach it to
            // T_b, just move on to the next sample.
            _ => {}
        }

        mem::swap(&mut t_a, &mut t_b);
    }

    merge_graphs(t_a, t_b, None)
}

fn merge_graphs<S: Clone>(
    a: UnGraph<S, f64>,
    b: UnGraph<S, f64>,
    fulcrum: Option<(NodeIndex, NodeIndex, f64)>,
) -> UnGraph<S, f64> {
    let mut c = a;
    let mut b_c_map = HashMap::with_capacity(b.node_count());

    // Copy the nodes from b into c, keeping track of the mapping from their old
    // IDs to their new ones.
    for (n_b, s) in b.node_references() {
        let n_c = c.add_node(s.clone());
        b_c_map.insert(n_b, n_c);
    }

    // Copy edges from b to c, using the ID translation map as a guide.
    for e_b in b.edge_references() {
        let source_c = b_c_map[&e_b.source()];
        let target_c = b_c_map[&e_b.target()];

        c.add_edge(source_c, target_c, *e_b.weight());
    }

    // If a fulcrum node exists to connect the two graphs then add it to the
    // combined graph.
    if let Some((q_a, q_b, dist)) = fulcrum {
        c.add_edge(q_a, b_c_map[&q_b], dist);
    }

    c
}
