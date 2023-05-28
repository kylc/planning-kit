use crate::state::StateSpace;

pub trait StateValidator {
    type State;

    fn validate_state(&self, s: &Self::State) -> bool;
}

pub trait MotionValidator: StateValidator {
    fn validate_motion(&self, a: &Self::State, b: &Self::State) -> bool;
}

/// Validates motions by validating states at discrete intervals interpolated
/// from the start to end state.
pub struct DiscreteMotionValidator<S, V> {
    /// The space to be used for interpolation and distance measurement.
    pub space: S,

    /// The underlying validator to be used on individual states.
    pub base_validator: V,

    /// The state space distance at which validation samples will be taken
    /// between the start and end points of the motion.
    pub discretization: f64,
}

impl<S, V> DiscreteMotionValidator<S, V>
where
    S: StateSpace,
    V: StateValidator<State = S::State>,
{
    pub fn new(space: S, base_checker: V, discretization: f64) -> Self {
        assert!(discretization >= 0.0);

        Self {
            space,
            base_validator: base_checker,
            discretization,
        }
    }
}

impl<S, V> StateValidator for DiscreteMotionValidator<S, V>
where
    S: StateSpace,
    V: StateValidator<State = S::State>,
{
    type State = <S as StateSpace>::State;

    fn validate_state(&self, s: &Self::State) -> bool {
        self.base_validator.validate_state(s)
    }
}

impl<S, V> MotionValidator for DiscreteMotionValidator<S, V>
where
    S: StateSpace,
    V: StateValidator<State = S::State>,
{
    fn validate_motion(&self, a: &Self::State, b: &Self::State) -> bool {
        let steps = self.space.distance(a, b) / self.discretization;
        let steps = (steps as usize).max(2); // must at least check the endpoints

        (0..steps).all(|i| {
            let t = i as f64 / (steps - 1) as f64;
            let s = self.space.interpolate(a, b, t);

            self.base_validator.validate_state(&s)
        })
    }
}
