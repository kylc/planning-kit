use nalgebra::{Complex, DMatrix, DVector};

use crate::constraint::real::Constraint;

trait ComplexConstraint: Send {
    fn n_ambient(&self) -> usize;

    fn n_co(&self) -> usize;

    fn position(&self, state: &DVector<Complex<f64>>) -> DVector<Complex<f64>>;

    fn jacobian(&self, state: &DVector<f64>, h: f64) -> DMatrix<f64> {
        let mut jac = DMatrix::zeros(self.n_co(), self.n_ambient());

        // Partial derivatives are computed by Complex-Step Differentiation,
        // which is not prone to numerical error.
        //
        // ∂f/∂x ≈ Im[f(x + ih) / h]
        for i in 0..self.n_ambient() {
            let mut cplx_state = into_complex(state.clone());
            cplx_state[i].im = h;
            let cplx_f = self.position(&cplx_state);

            let col = imag_part(cplx_f) / h;
            jac.set_column(i, &col);
        }

        jac
    }

    fn clone_box(&self) -> Box<dyn Constraint>;
}

impl<T> Constraint for T
where
    T: ComplexConstraint,
{
    fn n_ambient(&self) -> usize {
        self.n_ambient()
    }

    fn n_co(&self) -> usize {
        self.n_co()
    }

    fn position(&self, state: &DVector<f64>) -> DVector<f64> {
        let cplx_state = into_complex(state.clone());
        let cplx_pos = self.position(&cplx_state);
        real_part(cplx_pos)
    }

    fn jacobian(&self, state: &DVector<f64>, h: f64) -> DMatrix<f64> {
        self.jacobian(state, h)
    }

    fn clone_box(&self) -> Box<dyn Constraint> {
        self.clone_box()
    }
}

fn into_complex(v: DVector<f64>) -> DVector<Complex<f64>> {
    nalgebra::convert(v)
}

fn real_part(v: DVector<Complex<f64>>) -> DVector<f64> {
    DVector::from_iterator(v.len(), v.into_iter().map(|c| c.re))
}

fn imag_part(v: DVector<Complex<f64>>) -> DVector<f64> {
    DVector::from_iterator(v.len(), v.into_iter().map(|c| c.im))
}
