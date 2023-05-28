use std::ops::Deref;

use nalgebra::DVector;
use pyo3::{exceptions::PyNotImplementedError, prelude::*};

use crate::{constraint::PyConstraint, util::extract_vector};
use planning_kit::{
    constraint::ProjectedEuclideanSpace,
    state::{DynamicEuclideanSpace, StateSpace},
    util::newton_raphson::NewtonRaphsonOpts,
};

#[pyclass]
#[pyo3(name = "StateSpace")]
#[derive(Clone)]
// TODO: Hide the inner variant
pub struct PyStateSpace(pub SpaceVariant);

impl Deref for PyStateSpace {
    type Target = SpaceVariant;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl PyStateSpace {
    fn as_euclidean(&self) -> Option<&DynamicEuclideanSpace> {
        match &self.0 {
            SpaceVariant::Euclidean(s) => Some(s),
            _ => None,
        }
    }

    fn as_projected(&self) -> Option<&ProjectedEuclideanSpace> {
        match &self.0 {
            SpaceVariant::ProjectedEuclidean(s) => Some(s),
            _ => None,
        }
    }
}

#[pymethods]
impl PyStateSpace {
    #[staticmethod]
    pub fn euclidean(lo: &PyAny, hi: &PyAny) -> PyResult<Self> {
        let lo = DVector::from_vec(lo.extract()?);
        let hi = DVector::from_vec(hi.extract()?);
        Ok(Self(SpaceVariant::Euclidean(DynamicEuclideanSpace::new(
            lo, hi,
        ))))
    }

    #[staticmethod]
    pub fn projected(ambient: PyStateSpace, constraint: PyConstraint) -> PyResult<Self> {
        let ambient = ambient
            .as_euclidean()
            .ok_or(PyNotImplementedError::new_err(
                "non-Euclidean ambient spaces not supported",
            ))?;
        Ok(PyStateSpace(SpaceVariant::ProjectedEuclidean(
            ProjectedEuclideanSpace {
                ambient: ambient.clone(),
                constraint: Box::new(constraint),
                opts: NewtonRaphsonOpts::default(),
            },
        )))
    }

    #[staticmethod]
    pub fn custom(object: PyObject) -> PyResult<Self> {
        Ok(Self(SpaceVariant::Custom(object)))
    }

    pub fn sample_uniform(&self) -> Option<Vec<f64>> {
        let state = StateSpace::sample_uniform(&self.0)?;
        Some(state.as_slice().to_vec())
    }

    pub fn project(&self, state: &PyAny) -> PyResult<Option<Vec<f64>>> {
        let state = extract_vector(state)?;
        let space = self.as_projected().ok_or(PyNotImplementedError::new_err(
            "projection is only supported on ProjectedEuclideanSpace",
        ))?;

        let s_proj = space
            .constraint
            .project(&state, &NewtonRaphsonOpts::default());
        Ok(s_proj.map(|s| s.as_slice().to_vec()))
    }
}

#[derive(Clone)]
pub enum SpaceVariant {
    Euclidean(DynamicEuclideanSpace),
    ProjectedEuclidean(ProjectedEuclideanSpace),
    Custom(PyObject),
}

impl StateSpace for SpaceVariant {
    type State = DVector<f64>;

    fn sample_uniform(&self) -> Option<Self::State> {
        match self {
            SpaceVariant::Euclidean(s) => s.sample_uniform(),
            SpaceVariant::ProjectedEuclidean(s) => s.sample_uniform(),
            SpaceVariant::Custom(obj) => Python::with_gil(|py| {
                let result = obj.call_method0(py, "sample_uniform").unwrap();
                let result = extract_vector(result.as_ref(py)).unwrap();

                Some(result)
            }),
        }
    }

    fn sample_uniform_near(&self, near: &Self::State, dist: f64) -> Option<Self::State> {
        match self {
            SpaceVariant::Euclidean(s) => s.sample_uniform_near(near, dist),
            SpaceVariant::ProjectedEuclidean(s) => s.sample_uniform_near(near, dist),
            SpaceVariant::Custom(obj) => Python::with_gil(|py| {
                let args = (near.as_slice().to_vec(), dist);
                let result = obj.call_method1(py, "sample_uniform_near", args).unwrap();
                let result = extract_vector(result.as_ref(py)).unwrap();

                Some(result)
            }),
        }
    }

    fn distance(&self, a: &Self::State, b: &Self::State) -> f64 {
        match self {
            SpaceVariant::Euclidean(s) => s.distance(a, b),
            SpaceVariant::ProjectedEuclidean(s) => s.distance(a, b),
            SpaceVariant::Custom(obj) => Python::with_gil(|py| {
                let args = (a.as_slice().to_vec(), b.as_slice().to_vec());
                let result = obj.call_method1(py, "distance", args).unwrap();

                result.extract::<f64>(py).unwrap()
            }),
        }
    }

    fn interpolate(&self, a: &Self::State, b: &Self::State, alpha: f64) -> Self::State {
        match self {
            SpaceVariant::Euclidean(s) => s.interpolate(a, b, alpha),
            SpaceVariant::ProjectedEuclidean(s) => s.interpolate(a, b, alpha),
            SpaceVariant::Custom(obj) => Python::with_gil(|py| {
                let args = (a.as_slice().to_vec(), b.as_slice().to_vec(), alpha);
                let result = obj.call_method1(py, "interpolate", args).unwrap();

                extract_vector(result.as_ref(py)).unwrap()
            }),
        }
    }
}
