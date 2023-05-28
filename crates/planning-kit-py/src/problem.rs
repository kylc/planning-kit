use nalgebra::{DVector, Point3};
use parry3d_f64::shape::TriMesh;
use planning_kit::{util::kinematics::KinematicChain, validator::StateValidator};
use pyo3::prelude::*;

#[pyclass]
#[derive(Clone)]
pub struct Mesh {
    vertices: Vec<f64>,
    indices: Vec<u32>,
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
#[derive(Clone)]
pub struct KinematicChainProblem {
    inner: planning_kit::problem::KinematicChainProblem,
}

#[pymethods]
impl KinematicChainProblem {
    #[new]
    pub fn from_urdf_string(s: &str, environment: Mesh) -> PyResult<KinematicChainProblem> {
        let robot = urdf_rs::read_from_string(s).unwrap();
        let chain = KinematicChain::from_urdf(&robot);
        let problem = planning_kit::problem::KinematicChainProblem::new(chain, environment.into());

        Ok(KinematicChainProblem { inner: problem })
    }

    pub fn nq(&self) -> usize {
        self.inner.nq()
    }
}

impl StateValidator for KinematicChainProblem {
    type State = DVector<f64>;

    fn validate_state(&self, s: &Self::State) -> bool {
        self.inner.validate_state(s)
    }
}
