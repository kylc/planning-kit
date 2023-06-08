use nalgebra::DVector;
use pyo3::prelude::*;

pub fn extract_vector(any: &PyAny) -> PyResult<DVector<f64>> {
    let vec: Vec<f64> = any.extract()?;
    Ok(DVector::from_row_slice(&vec))
}

pub fn wrap_nparray(list: &PyObject) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        let np = PyModule::import(py, "numpy")?;
        let asarray = np.getattr("asarray")?;
        let array = asarray.call1((list.into_py(py),))?;

        Ok(array.into_py(py))
    })
}
