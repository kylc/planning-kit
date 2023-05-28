use nalgebra::DVector;
use pyo3::prelude::*;

pub fn extract_vector(any: &PyAny) -> PyResult<DVector<f64>> {
    let vec: Vec<f64> = any.extract()?;
    Ok(DVector::from_row_slice(&vec))
}
