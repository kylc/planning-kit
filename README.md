# planning-kit

<p>
    <img alt="MIT"    src="https://img.shields.io/badge/license-MIT-blue.svg">
    <img alt="Apache" src="https://img.shields.io/badge/license-Apache-blue.svg">
    <img alt="CI"     src="https://github.com/kylc/planning-kit/actions/workflows/ci.yaml/badge.svg">
</p>

planning-kit is an experimental toolbox implementing sampling-based motion planning for high-dimensional systems. The toolbox is exposed in the form of a Rust library with Python bindings.

## Usage

Plan around a sphere obstacle in $\mathbb{R}^3$ using the RRT-Connect planner:

https://github.com/kylc/planning-kit/blob/e83075cf0f9afa06b3d94a9de964c3397302c08e/demo/01_intro.py#L1-L33

## Details

### State Space

A state space (or configuration space) is defined by all possible configurations of the system. Arbitrary state space definitions are possible for use with the library.

For example, to define a Euclidean space (which has equivalent behavior to the built-in `StateSpace.euclidean`):

https://github.com/kylc/planning-kit/blob/e83075cf0f9afa06b3d94a9de964c3397302c08e/demo/02_custom_space.py#L1-L26

### Constraints

Planning subject to manifold constraints is implemented via the unifying framework IMACS (implicit manifold configuration space) presented in [[KiMK19]](#KiMK19). Projection is used to adhere samples from an ambient space to the constrained manifold while preserving the planner properties of probabilistic completeness and asymptotic optimality.

In order to converge on an adhering sample, Newton's method is used to approximate the roots of the constraints. Constraint Jacobians (required for Newton's method) are automatically computed via central finite difference or complex-step differentiation, or are provided analytically.

For example, we can define a constraint which confines the state space to the surface of a unit sphere. With this constraint, we can define a state space which is automatically projected into the manifold. This state space can be used like any other, e.g. for use in motion planning algorithms.

https://github.com/kylc/planning-kit/blob/e83075cf0f9afa06b3d94a9de964c3397302c08e/demo/03_constraints.py#L1-L14

### Planning

| Planner     | Category     | Optimal | Reference           |
|-------------|--------------|---------|---------------------|
| PRM         | Multi-query  | :x:     | [[KSLO96]](#KSLO96) |
| RRT         | Single-query | :x:     | [[LaVa98]](#LaVa98) |
| RRT-Connect | Single-query | :x:     | [[KuLa00]](#KuLa00) |

### Post-processing

TODO

## References

<a id="KiMK19">[KiMK19]:</a>
Z. Kingston, M. Moll, and L. E. Kavraki, "Exploring implicit spaces for constrained sampling-based planning," The International Journal of Robotics Research, vol. 38, no. 10–11, pp. 1151–1178, Sep. 2019, doi: 10.1177/0278364919868530.

<a id="KSLO96">[KSLO96]:</a>
L. E. Kavraki, P. Svestka, J.-C. Latombe, and M. H. Overmars, "Probabilistic roadmaps for path planning in high-dimensional configuration spaces," IEEE Transactions on Robotics and Automation, vol. 12, no. 4, pp. 566–580, Aug. 1996, doi: 10.1109/70.508439.

<a id="LaVa98">[LaVa98]:</a>
S. LaValle, "Rapidly-exploring random trees : a new tool for path planning," The annual research report, 1998, Accessed: May 26, 2023. [Online]. Available: https://www.semanticscholar.org/paper/Rapidly-exploring-random-trees-%3A-a-new-tool-for-LaValle/d967d9550f831a8b3f5cb00f8835a4c866da60ad

<a id="KuLa00">[KuLa00]:</a>
J. J. Kuffner and S. M. LaValle, "RRT-connect: An efficient approach to single-query path planning," in Proceedings 2000 ICRA. Millennium Conference. IEEE International Conference on Robotics and Automation. Symposia Proceedings (Cat. No.00CH37065), San Francisco, CA, USA: IEEE, 2000, pp. 995–1001. doi: 10.1109/ROBOT.2000.844730.
