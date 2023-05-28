use nalgebra::{allocator::Allocator, Const, DefaultAllocator, Dim, Dyn, OVector, U1};
use rand::{distributions::Uniform, Rng};

use crate::state::StateSpace;

/// A Euclidean space (see [GenericEuclideanSpace]) with a compile-time
/// dimension allowing for static memory allocation.
pub type EuclideanSpace<const N: usize> = GenericEuclideanSpace<Const<N>>;

/// A Euclidean space (see [GenericEuclideanSpace]) with dimension known only at
/// runtime, allowing for spaces of unknown compile-time dimension.
pub type DynamicEuclideanSpace = GenericEuclideanSpace<Dyn>;

#[derive(Clone, Debug)]
pub struct GenericEuclideanSpace<D: Dim>
where
    DefaultAllocator: Allocator<f64, D>,
{
    dim: usize,
    lo: OVector<f64, D>,
    hi: OVector<f64, D>,
}

impl<D: Dim> GenericEuclideanSpace<D>
where
    DefaultAllocator: Allocator<f64, D>,
{
    pub fn new(lo: OVector<f64, D>, hi: OVector<f64, D>) -> Self {
        assert!(lo.shape() == hi.shape());
        assert!(lo <= hi);

        Self {
            dim: lo.len(),
            lo,
            hi,
        }
    }
}

impl<D: Dim> StateSpace for GenericEuclideanSpace<D>
where
    DefaultAllocator: Allocator<f64, D>,
{
    type State = OVector<f64, D>;

    fn sample_uniform(&self) -> Option<Self::State> {
        let mut rng = rand::thread_rng();

        let scale = &self.hi - &self.lo;
        let dist = Uniform::new(0.0, 1.0);
        Some(
            scale.component_mul(&OVector::from_fn_generic(
                Dim::from_usize(self.dim),
                U1,
                |_, _| rng.sample(dist),
            )) + &self.lo,
        )
    }

    fn sample_uniform_near(&self, near: &Self::State, dist: f64) -> Option<Self::State> {
        assert!(dist >= 0.0);

        let mut rng = rand::thread_rng();

        let dist = Uniform::new(-dist, dist);
        Some(OVector::from_fn_generic(
            Dim::from_usize(self.dim),
            U1,
            |i, _| near[i] + rng.sample(dist),
        ))
    }

    fn distance(&self, a: &Self::State, b: &Self::State) -> f64 {
        a.metric_distance(b)
    }

    fn interpolate(&self, a: &Self::State, b: &Self::State, alpha: f64) -> Self::State {
        assert!(alpha >= 0.0);
        assert!(alpha <= 1.0);

        a.lerp(b, alpha)
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::Vector2;

    use super::*;

    #[test]
    fn test_euclidean_bounds() {
        let lo = Vector2::new(0.0, 0.0);
        let hi = Vector2::new(10.0, 10.0);
        let space = EuclideanSpace::new(lo, hi);

        for _ in 0..50 {
            let state = space.sample_uniform().unwrap();
            assert!(lo <= state);
            assert!(state <= hi);
        }
    }

    #[test]
    fn test_euclidean_near() {
        let lo = Vector2::new(0.0, 0.0);
        let hi = Vector2::new(1.0, 1.0);
        let space = EuclideanSpace::new(lo, hi);

        let near = Vector2::new(0.5, 0.5);
        let dist = 0.1;
        for _ in 0..50 {
            let state = space.sample_uniform_near(&near, dist).unwrap();
            assert!((state - near).abs() < Vector2::repeat(dist));
        }
    }
}
