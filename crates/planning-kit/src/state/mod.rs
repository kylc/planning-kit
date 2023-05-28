mod euclidean;

pub use euclidean::*;

pub trait StateSpace {
    type State;

    fn sample_uniform(&self) -> Option<Self::State>;

    fn sample_uniform_near(&self, near: &Self::State, dist: f64) -> Option<Self::State>;

    fn distance(&self, a: &Self::State, b: &Self::State) -> f64;

    fn interpolate(&self, a: &Self::State, b: &Self::State, alpha: f64) -> Self::State;
}
