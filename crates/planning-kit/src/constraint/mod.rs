use nalgebra::DVector;

use crate::{
    constraint::real::Constraint,
    state::{DynamicEuclideanSpace, StateSpace},
    util::newton_raphson::NewtonRaphsonOpts,
};

pub mod complex;
pub mod real;

#[derive(Clone)]
pub struct ProjectedEuclideanSpace {
    pub constraint: Box<dyn Constraint>,
    pub ambient: DynamicEuclideanSpace,
    pub opts: NewtonRaphsonOpts,
}

impl StateSpace for ProjectedEuclideanSpace {
    type State = DVector<f64>;

    fn sample_uniform(&self) -> Option<Self::State> {
        let ambient_state = self.ambient.sample_uniform()?;
        // TODO: Does not guarantee that projected samples will be uniformly
        // distributed.
        self.constraint.project(&ambient_state, &self.opts)
    }

    fn sample_uniform_near(&self, near: &Self::State, dist: f64) -> Option<Self::State> {
        let ambient_state = self.ambient.sample_uniform_near(near, dist)?;
        // TODO: The projection operator may push the sample outside of `dist.`
        self.constraint.project(&ambient_state, &self.opts)
    }

    fn distance(&self, a: &Self::State, b: &Self::State) -> f64 {
        // TODO: account for the geodesics
        self.ambient.distance(a, b)
    }

    fn interpolate(&self, a: &Self::State, b: &Self::State, alpha: f64) -> Self::State {
        let ambient_state = self.ambient.interpolate(a, b, alpha);
        self.constraint
            .project(&ambient_state, &self.opts)
            .unwrap_or_else(|| a.clone()) // TODO
    }
}
