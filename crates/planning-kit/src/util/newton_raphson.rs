use nalgebra::{
    allocator::Allocator, DefaultAllocator, Dim, DimDiff, DimMin, DimMinimum, DimSub, OMatrix,
    OVector, U1,
};

#[derive(Clone, Debug)]
pub struct NewtonRaphsonOpts {
    pub step_size: f64,

    /// A constraint is considered adhered to when ||F(q)|| < tolerance.
    pub tolerance: f64,
    pub max_iter: usize,
}

impl Default for NewtonRaphsonOpts {
    fn default() -> Self {
        Self {
            step_size: 0.001,
            tolerance: 0.01,
            max_iter: 100,
        }
    }
}

pub fn newton_raphson<F, J, R: Dim, C: Dim>(
    x_init: &OVector<f64, C>,
    f_func: F,
    jac_func: J,
    opts: &NewtonRaphsonOpts,
) -> Option<OVector<f64, C>>
where
    R: DimMin<C>,
    DimMinimum<R, C>: DimSub<U1>, // for Bidiagonal.
    DefaultAllocator: Allocator<f64, R, C>
        + Allocator<f64, C>
        + Allocator<f64, R>
        + Allocator<f64, DimDiff<DimMinimum<R, C>, U1>>
        + Allocator<f64, DimMinimum<R, C>, C>
        + Allocator<f64, R, DimMinimum<R, C>>
        + Allocator<f64, DimMinimum<R, C>>
        + Allocator<(usize, usize), DimMinimum<R, C>>
        + Allocator<(f64, usize), DimMinimum<R, C>>,

    F: Fn(&OVector<f64, C>) -> OVector<f64, R>,
    J: Fn(&OVector<f64, C>, f64) -> OMatrix<f64, R, C>,
{
    let mut x_new = x_init.clone();

    // Solve for the simultaneous zeros of the constraint by Newton's
    // method. We iteratively converge to the zero by applying this
    // expression:
    //
    // x_{n+1} = x_n - delta_x
    // delta_x = J(x_n)^{-1} * F(x_n)
    for _ in 0..opts.max_iter {
        let x = f_func(&x_new);
        let j = jac_func(&x_new, opts.step_size);

        // Solve `J * x = f` for x (aka `x = J^{-1} * f`)
        // TODO: Think about the `eps` argument to svd.solve() and how it
        // relates to the convergence tolerance.
        let svd = j.svd(true, true);
        let delta_x = svd.solve(&x, opts.tolerance).unwrap();

        // Improve our estimate of the projected state.
        x_new -= delta_x;

        // Check for convergence
        if x.norm() < opts.tolerance {
            return Some(x_new);
        }
    }

    None
}
