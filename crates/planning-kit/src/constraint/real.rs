use nalgebra::{DMatrix, DVector};

use crate::util::newton_raphson::{newton_raphson, NewtonRaphsonOpts};

// TODO: For now this module uses dynamically-allocated vectors only, rather
// than generic ones. This is because the type constraints for the required
// operations are difficult to express for static vectors.

/// A composite manifold constraint function F(q) which is adhered when F(q) =
/// 0. F(q) is assumed to be C2-smooth.
pub trait Constraint: Send {
    fn n_ambient(&self) -> usize;

    fn n_co(&self) -> usize;

    fn position(&self, state: &DVector<f64>) -> DVector<f64>;

    fn jacobian(&self, state: &DVector<f64>, h: f64) -> DMatrix<f64> {
        // Default implementation is to compute the Jacobian numerically using
        // simple central finite differences. Implementers of this trait may
        // choose to override this method, perhaps to compute the Jacobian
        // analytically.
        // NOTE: OMPL uses a five-point stencil here for more accuracy.
        let mut jac = DMatrix::zeros(self.n_co(), self.n_ambient());

        // Compute [ ∂f/∂x_1 ... ∂f/∂x_n ]
        for i in 0..self.n_ambient() {
            let mut off = DVector::zeros(self.n_ambient());
            off[i] = h / 2.0;

            let x1 = state - &off;
            let x2 = state + &off;

            let y1 = self.position(&x1);
            let y2 = self.position(&x2);

            let dfdx = (y2 - y1) / h;
            jac.set_column(i, &dfdx);
        }

        jac
    }

    fn project(&self, state: &DVector<f64>, opts: &NewtonRaphsonOpts) -> Option<DVector<f64>> {
        newton_raphson(
            state,
            |x| self.position(x),
            |x, h| self.jacobian(x, h),
            opts,
        )
    }

    fn clone_box(&self) -> Box<dyn Constraint>;
}

impl Clone for Box<dyn Constraint> {
    fn clone(&self) -> Box<dyn Constraint> {
        self.clone_box()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The surface of an n-sphere centered at the origin.
    #[derive(Clone)]
    pub struct NumericalSphere<const N: usize>(f64);

    impl<const N: usize> Constraint for NumericalSphere<N> {
        fn n_ambient(&self) -> usize {
            N
        }

        fn n_co(&self) -> usize {
            1
        }

        fn position(&self, state: &DVector<f64>) -> DVector<f64> {
            DVector::from_row_slice(&[state.norm_squared() - 1.0])
        }

        fn clone_box(&self) -> Box<dyn Constraint> {
            Box::new(self.clone())
        }
    }

    // The same constraint as above but with analytical partial derivatives.
    #[derive(Clone)]
    pub struct AnalyticalSphere<const N: usize>(f64);

    impl<const N: usize> Constraint for AnalyticalSphere<N> {
        fn n_ambient(&self) -> usize {
            N
        }

        fn n_co(&self) -> usize {
            1
        }

        fn position(&self, state: &DVector<f64>) -> DVector<f64> {
            DVector::from_row_slice(&[state.norm_squared() - 1.0])
        }

        fn jacobian(&self, state: &DVector<f64>, _h: f64) -> DMatrix<f64> {
            let sdot = 2.0 * state;
            DMatrix::from_rows(&[sdot.transpose()])
        }

        fn clone_box(&self) -> Box<dyn Constraint> {
            Box::new(self.clone())
        }
    }

    #[test]
    fn test_numerical_diff() {
        const N: usize = 3;
        let numerical = NumericalSphere::<N>(1.0);
        let analytical = AnalyticalSphere::<N>(1.0);

        let h = 0.0001;
        let tol = 10.0 * h;
        for _ in 0..100 {
            let s = DVector::new_random(N);

            let j_n = numerical.jacobian(&s, h);
            let j_a = analytical.jacobian(&s, h);

            assert!((j_n - &j_a).norm() < tol);
        }
    }

    #[test]
    fn test_analytical_projection() {
        const N: usize = 3;
        let constraint = AnalyticalSphere::<N>(1.0);

        let opts = NewtonRaphsonOpts::default();
        for _ in 0..100 {
            // q = project(p)
            let p = 5.0 * DVector::new_random(N);
            let q = constraint.project(&p, &opts);
            assert!(q.is_some());

            // f(q) <= tol
            assert!(constraint.position(&q.unwrap()).norm() <= opts.tolerance);
        }
    }
}
