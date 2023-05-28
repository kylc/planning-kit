use nalgebra::DVector;
use planning_kit::{constraint::real::Constraint, util::newton_raphson::NewtonRaphsonOpts};
use pyo3::{prelude::*, types::PyTuple};

use crate::util::extract_vector;

#[pyclass]
#[pyo3(name = "Constraint")]
#[derive(Clone)]
pub struct PyConstraint {
    n_ambient: usize,
    n_co: usize,
    position_cb: PyObject,
}

#[pymethods]
impl PyConstraint {
    #[new]
    #[pyo3(signature = (n_ambient, n_co, position_cb))]
    pub fn new(n_ambient: usize, n_co: usize, position_cb: PyObject) -> Self {
        Self {
            n_ambient,
            n_co,
            position_cb,
        }
    }

    pub fn project(&self, state: &PyAny) -> PyResult<Vec<f64>> {
        let state = extract_vector(state)?;

        Ok(
            Constraint::project(self, &state, &NewtonRaphsonOpts::default())
                .unwrap()
                .as_slice()
                .to_vec(),
        )
    }
}

impl Constraint for PyConstraint {
    fn n_ambient(&self) -> usize {
        self.n_ambient
    }

    fn n_co(&self) -> usize {
        self.n_co
    }

    fn position(&self, state: &nalgebra::DVector<f64>) -> nalgebra::DVector<f64> {
        Python::with_gil(|py| {
            let args = PyTuple::new(py, [state.as_slice().to_owned()]);
            let val = self.position_cb.call1(py, args).unwrap();
            let res = val.extract::<Vec<f64>>(py).unwrap();
            DVector::from_row_slice(&res)
        })
    }

    fn clone_box(&self) -> Box<dyn Constraint> {
        Box::new(self.clone())
    }
}
