use constraint::PyConstraint;
use nalgebra::DVector;
use planning_kit::{planner::prm::Roadmap, state::StateSpace, validator::MotionValidator};
use problem::{PyKinematicChainProblem, Mesh};
use pyo3::{prelude::*, types::PyTuple};

use state::PyStateSpace;
use util::extract_vector;

pub mod constraint;
pub mod problem;
pub mod state;
pub mod util;

pub struct PyStateValidator {
    callable: PyObject,
}

impl planning_kit::validator::StateValidator for PyStateValidator {
    type State = DVector<f64>;

    fn validate_state(&self, s: &Self::State) -> bool {
        Python::with_gil(|py| {
            let args = PyTuple::new(py, [s.as_slice()]);
            let val = self.callable.call1(py, args).unwrap();
            val.extract::<bool>(py).unwrap()
        })
    }
}

#[pyclass]
#[pyo3(name = "Roadmap")]
pub struct PyRoadmap {
    inner: Roadmap<DVector<f64>>,
}

#[pymethods]
impl PyRoadmap {
    pub fn num_nodes(&self) -> usize {
        self.inner.graph.node_count()
    }

    pub fn num_edges(&self) -> usize {
        self.inner.graph.edge_count()
    }

    pub fn points(&self) -> PyObject {
        Python::with_gil(|py| {
            self.inner
                .graph
                .node_weights()
                .map(|v| v.as_slice())
                .collect::<Vec<_>>()
                .to_object(py)
        })
    }

    pub fn shortest_path(
        &self,
        space: &PyStateSpace,
        start: &PyAny,
        goal: &PyAny,
    ) -> PyResult<Option<PyPath>> {
        let start = extract_vector(start)?;
        let goal = extract_vector(goal)?;

        let path = self.inner.query(&space.0, &start, &goal);
        Ok(path.map(|p| PyPath {
            inner: p.into_iter().collect(),
        }))
    }
}

#[pyclass]
#[pyo3(name = "Path")]
pub struct PyPath {
    inner: Vec<DVector<f64>>,
}

#[pymethods]
impl PyPath {
    pub fn points(&self) -> Vec<Vec<f64>> {
        self.inner
            .iter()
            .map(|v| v.as_slice().to_owned())
            .collect::<Vec<_>>()
    }

    pub fn shortcut(
        &self,
        space: &PyStateSpace,
        validator: PyObject,
        discretization: f64,
    ) -> PyPath {
        let validator = PyStateValidator {
            callable: validator,
        };
        let validator = planning_kit::validator::DiscreteMotionValidator::new(
            space.0.clone(),
            validator,
            discretization,
        );

        let attempt = |input_path: &Vec<DVector<f64>>| -> Vec<DVector<f64>> {
            let mut shortcut = vec![];

            for abc in input_path.chunks(3) {
                if abc.len() < 3 {
                    shortcut.extend_from_slice(abc);
                } else if validator.validate_motion(&abc[0], &abc[2]) {
                    shortcut.push(abc[0].clone());
                    shortcut.push(abc[2].clone());
                } else {
                    shortcut.push(abc[0].clone());
                    shortcut.push(abc[1].clone());
                    shortcut.push(abc[2].clone());
                }
            }

            shortcut
        };

        let mut shortcut_path = self.inner.clone();
        loop {
            let shorter_path = attempt(&shortcut_path);

            if shorter_path.len() < shortcut_path.len() {
                shortcut_path = shorter_path;
            } else {
                break;
            }
        }
        PyPath {
            inner: shortcut_path,
        }
    }

    pub fn interpolate(&self, space: &PyStateSpace, t: f64) -> Vec<f64> {
        let total_length: f64 = self
            .inner
            .windows(2)
            .map(|ab| space.distance(&ab[0], &ab[1]))
            .sum();
        let l = t * total_length;

        let mut distance = 0.0;
        for ab in self.inner.windows(2) {
            let next_distance = distance + space.distance(&ab[0], &ab[1]);

            if distance <= l && l <= next_distance {
                let inner_t = (l - distance) / (next_distance - distance);
                return space
                    .interpolate(&ab[0], &ab[1], inner_t)
                    .as_slice()
                    .to_owned();
            }

            distance = next_distance;
        }

        DVector::zeros(6).as_slice().to_owned()
    }
}

#[pyfunction]
#[pyo3(signature = (space, validator, /, n_samples=100, discretization=0.1))]
pub fn prm_roadmap(
    space: &PyStateSpace,
    validator: PyObject,
    n_samples: usize,
    discretization: f64,
) -> PyRoadmap {
    let validator = PyStateValidator {
        callable: validator,
    };
    let validator = planning_kit::validator::DiscreteMotionValidator::new(
        space.0.clone(),
        validator,
        discretization,
    );

    PyRoadmap {
        inner: planning_kit::planner::prm::build_roadmap(&space.0, &validator, n_samples, 10),
    }
}

#[pyfunction]
#[pyo3(signature = (space, validator, /, start, goal, discretization=0.1, steering_dist=1.0))]
pub fn rrt(
    space: &PyStateSpace,
    validator: PyObject,
    start: &PyAny,
    goal: &PyAny,
    discretization: f64,
    steering_dist: f64,
) -> PyResult<PyRoadmap> {
    let start = DVector::from_vec(start.extract()?);
    let goal = DVector::from_vec(goal.extract()?);

    let validator = PyStateValidator {
        callable: validator,
    };
    let validator = planning_kit::validator::DiscreteMotionValidator::new(
        space.0.clone(),
        validator,
        discretization,
    );

    Ok(PyRoadmap {
        inner: Roadmap {
            graph: planning_kit::planner::rrt::explore(
                &space.0,
                &validator,
                steering_dist,
                start,
                goal,
            ),
        },
    })
}

#[pyfunction]
#[pyo3(signature = (space, validator, /, start, goal, discretization=0.1, steering_dist=1.0, n=1000))]
pub fn rrt_connect(
    space: &PyStateSpace,
    validator: PyObject,
    start: &PyAny,
    goal: &PyAny,
    discretization: f64,
    steering_dist: f64,
    n: usize,
) -> PyResult<PyRoadmap> {
    let start = DVector::from_vec(start.extract()?);
    let goal = DVector::from_vec(goal.extract()?);

    let validator = PyStateValidator {
        callable: validator,
    };
    let validator = planning_kit::validator::DiscreteMotionValidator::new(
        space.0.clone(),
        validator,
        discretization,
    );

    Ok(PyRoadmap {
        inner: Roadmap {
            graph: planning_kit::planner::rrt_connect::explore(
                &space.0,
                &validator,
                steering_dist,
                start,
                goal,
                n,
            ),
        },
    })
}

#[pyfunction]
#[pyo3(signature = (space, problem, /, discretization=0.1, n_samples=1000, connectivity=10))]
pub fn prm_roadmap_problem(
    space: &PyStateSpace,
    problem: &PyKinematicChainProblem,
    discretization: f64,
    n_samples: usize,
    connectivity: usize,
) -> PyResult<PyRoadmap> {
    let validator = planning_kit::validator::DiscreteMotionValidator::new(
        space.0.clone(),
        problem.clone(),
        discretization,
    );

    Ok(PyRoadmap {
        inner: planning_kit::planner::prm::build_roadmap(
            &space.0,
            &validator,
            n_samples,
            connectivity,
        ),
    })
}

#[pyfunction]
#[pyo3(signature = (space, problem, /, start, goal, discretization=0.1, steering_dist=1.0))]
pub fn rrt_problem(
    space: &PyStateSpace,
    problem: &PyKinematicChainProblem,
    start: &PyAny,
    goal: &PyAny,
    discretization: f64,
    steering_dist: f64,
) -> PyResult<PyRoadmap> {
    let start = DVector::from_vec(start.extract()?);
    let goal = DVector::from_vec(goal.extract()?);

    let validator = planning_kit::validator::DiscreteMotionValidator::new(
        space.0.clone(),
        problem.clone(),
        discretization,
    );

    Ok(PyRoadmap {
        inner: Roadmap {
            graph: planning_kit::planner::rrt::explore(
                &space.0,
                &validator,
                steering_dist,
                start,
                goal,
            ),
        },
    })
}

#[pyfunction]
#[pyo3(signature = (space, problem, /, start, goal, discretization=0.1, steering_dist=1.0, n=1000))]
pub fn rrt_connect_problem(
    space: &PyStateSpace,
    problem: &PyKinematicChainProblem,
    start: &PyAny,
    goal: &PyAny,
    discretization: f64,
    steering_dist: f64,
    n: usize,
) -> PyResult<PyRoadmap> {
    let start = DVector::from_vec(start.extract()?);
    let goal = DVector::from_vec(goal.extract()?);

    let validator = planning_kit::validator::DiscreteMotionValidator::new(
        space.0.clone(),
        problem.clone(),
        discretization,
    );

    Ok(PyRoadmap {
        inner: Roadmap {
            graph: planning_kit::planner::rrt_connect::explore(
                &space.0,
                &validator,
                steering_dist,
                start,
                goal,
                n,
            ),
        },
    })
}

#[pymodule]
#[pyo3(name = "planning_kit")]
fn planning_kit_py(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyRoadmap>()?;
    m.add_class::<PyPath>()?;

    // constraint
    m.add_class::<PyConstraint>()?;

    // problem
    m.add_class::<Mesh>()?;
    m.add_class::<PyKinematicChainProblem>()?;

    // state
    m.add_class::<PyStateSpace>()?;

    m.add_function(wrap_pyfunction!(prm_roadmap, m)?)?;
    m.add_function(wrap_pyfunction!(prm_roadmap_problem, m)?)?;
    m.add_function(wrap_pyfunction!(rrt, m)?)?;
    m.add_function(wrap_pyfunction!(rrt_problem, m)?)?;
    m.add_function(wrap_pyfunction!(rrt_connect, m)?)?;
    m.add_function(wrap_pyfunction!(rrt_connect_problem, m)?)?;

    Ok(())
}
