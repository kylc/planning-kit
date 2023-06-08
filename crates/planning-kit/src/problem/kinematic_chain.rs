use std::f64::consts::PI;

use nalgebra::{DVector, Isometry3};
use parry3d_f64::{
    query::intersection_test,
    shape::{SharedShape, TriMesh},
};

use crate::{
    state::DynamicEuclideanSpace, util::kinematics::KinematicChain, validator::StateValidator,
};

#[derive(Clone)]
pub struct KinematicChainProblem {
    chain: KinematicChain,
    environment: SharedShape,
}

impl KinematicChainProblem {
    pub fn new(chain: KinematicChain, environment: TriMesh) -> Self {
        Self {
            chain,
            environment: SharedShape::new(environment),
        }
    }

    pub fn nq(&self) -> usize {
        self.chain.nq()
    }

    pub fn chain(&self) -> &KinematicChain {
        &self.chain
    }

    pub fn joint_space(&self) -> DynamicEuclideanSpace {
        // TODO: Use real joint limits from model
        let lo = DVector::repeat(self.nq(), -PI);
        let hi = DVector::repeat(self.nq(), PI);

        DynamicEuclideanSpace::new(lo, hi)
    }

    pub fn compute_link_tfm(&self, q: &[f64], link_name: &str) -> Option<Isometry3<f64>> {
        let (link_ix, _) = self
            .chain
            .links()
            .find(|(_, link)| link.name == link_name)?;

        let fk = self.chain.forward_kinematics(q);
        Some(fk.link_tfm(link_ix))
    }
}

impl StateValidator for KinematicChainProblem {
    type State = DVector<f64>;

    fn validate_state(&self, q: &Self::State) -> bool {
        let fk = self.chain.forward_kinematics(q.as_slice());

        // All links should be collision free with the environment.
        self.chain.links().all(|(link_ix, link)| {
            let in_collision = if let Some(link_shape) = &link.collision {
                let link_tfm = fk.link_tfm(link_ix);
                intersection_test(
                    &Isometry3::identity(),
                    &*self.environment,
                    &link_tfm,
                    link_shape,
                )
                .unwrap() // TODO: unwrap
            } else {
                // If there is no collision geometry, it cannot be in collision!
                false
            };

            !in_collision
        })
    }
}
