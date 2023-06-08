use std::collections::HashMap;

use nalgebra::{DVector, Point3};
use parry3d_f64::shape::TriMesh;
use planning_kit::{util::kinematics::KinematicChain, validator::StateValidator};
use pyo3::prelude::*;

use crate::util::{extract_vector, wrap_nparray};

#[pyclass]
#[derive(Clone)]
pub struct Mesh {
    #[pyo3(get)]
    pub vertices: Vec<f64>,

    #[pyo3(get)]
    pub indices: Vec<u32>,
}

#[pymethods]
impl Mesh {
    #[new]
    pub fn new(vertices: Vec<f64>, indices: Vec<u32>) -> Self {
        assert!(vertices.len() % 3 == 0);
        assert!(indices.len() % 3 == 0);

        Self { vertices, indices }
    }
}

impl From<Mesh> for TriMesh {
    fn from(mesh: Mesh) -> Self {
        let vertices = mesh.vertices.chunks(3).map(Point3::from_slice).collect();
        let indices = mesh
            .indices
            .chunks(3)
            .map(|xyz| [xyz[0], xyz[1], xyz[2]])
            .collect();
        TriMesh::new(vertices, indices)
    }
}

#[pyclass]
#[pyo3(name = "KinematicChainProblem")]
#[derive(Clone)]
pub struct PyKinematicChainProblem {
    inner: planning_kit::problem::KinematicChainProblem,
}

#[pymethods]
impl PyKinematicChainProblem {
    #[new]
    pub fn from_urdf_string(s: &str, environment: Mesh) -> PyResult<PyKinematicChainProblem> {
        let robot = urdf_rs::read_from_string(s).unwrap();
        let chain = KinematicChain::from_urdf(&robot);
        let problem = planning_kit::problem::KinematicChainProblem::new(chain, environment.into());

        Ok(PyKinematicChainProblem { inner: problem })
    }

    pub fn nq(&self) -> usize {
        self.inner.nq()
    }

    pub fn forward_kinematics(&self, q: &PyAny) -> PyResult<HashMap<String, PyObject>> {
        Python::with_gil(|py| {
            let q = extract_vector(q)?;
            let chain = self.inner.chain();
            let fk = chain.forward_kinematics(q.as_slice());

            let mut tfms = HashMap::new();
            for link in chain.links() {
                let link_tfm = fk.link_tfm(link.0).to_homogeneous();
                tfms.insert(
                    link.1.name.clone(),
                    wrap_nparray(
                        &[
                            [link_tfm.m11, link_tfm.m12, link_tfm.m13, link_tfm.m14],
                            [link_tfm.m21, link_tfm.m22, link_tfm.m23, link_tfm.m24],
                            [link_tfm.m31, link_tfm.m32, link_tfm.m33, link_tfm.m34],
                            [link_tfm.m41, link_tfm.m42, link_tfm.m43, link_tfm.m44],
                        ]
                        .into_py(py),
                    )?,
                );
            }

            Ok(tfms)
        })
    }
}

impl StateValidator for PyKinematicChainProblem {
    type State = DVector<f64>;

    fn validate_state(&self, s: &Self::State) -> bool {
        self.inner.validate_state(s)
    }
}
