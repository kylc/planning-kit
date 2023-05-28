use float_ord::FloatOrd;

use crate::state::StateSpace;

/// A brute force implementation of nearest neighbors.
///
/// Queries are evaluated by exhaustive comparison between all candidates in the
/// dataset. For N samples in D dimensions, this method scales as O(D*N^2).
///
/// This approach is competitive with more advanced algorithms for small
/// datasets.
pub fn nearest_neighbors_linear<'a, Space, I>(
    space: &Space,
    candidates: I,
    target: &Space::State,
    k: usize,
) -> Vec<(usize, f64)>
where
    Space: StateSpace + 'a,
    I: IntoIterator<Item = &'a Space::State>,
{
    let mut nearest = Vec::with_capacity(k);

    // Compute the distance from the target to every candidate, retaining
    // the nearest `k`.
    for (idx, candidate) in candidates.into_iter().enumerate() {
        let dist = space.distance(candidate, target);

        // Find where this distance falls within the list of nearest
        // candidates. If it would not reside in the top-k, then discard it.
        // Otherwise, add it to the list and truncate if necessary.
        match nearest.binary_search_by_key(&FloatOrd(dist), |&(_, d)| FloatOrd(d)) {
            Err(i) if i < k => {
                nearest.truncate(k - 1); // drop further to make room
                nearest.insert(i, (idx, dist));
            }
            _ => {}
        }
    }

    nearest
}

#[cfg(test)]
mod tests {
    use nalgebra::Vector2;

    use crate::state::EuclideanSpace;

    use super::*;

    #[test]
    fn test_linear_nearest_neighbor() {
        let space = EuclideanSpace::<2>::new(Vector2::zeros(), Vector2::zeros());

        let k0 = Vector2::new(0.0, 0.0);
        let k1 = Vector2::new(0.0, 0.01);
        let k2 = Vector2::new(0.02, 0.0);

        let j0 = Vector2::new(1.0, 0.0);
        let j1 = Vector2::new(1.0, 0.01);
        let j2 = Vector2::new(1.02, 0.0);

        let candidates = [k0, k1, k2, j0, j1, j2];
        let k0_nearest = nearest_neighbors_linear(&space, &candidates, &k0, 3);

        assert_eq!(k0_nearest[0].0, 0);
        assert_eq!(k0_nearest[1].0, 1);
        assert_eq!(k0_nearest[2].0, 2);
    }
}
