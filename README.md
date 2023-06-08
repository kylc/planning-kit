# planning-kit

<p>
    <img alt="MIT"    src="https://img.shields.io/badge/license-MIT-blue.svg">
    <img alt="Apache" src="https://img.shields.io/badge/license-Apache-blue.svg">
    <img alt="CI"     src="https://github.com/kylc/planning-kit/actions/workflows/ci.yaml/badge.svg">
</p>

planning-kit is an experimental toolbox implementing sampling-based motion planning for high-dimensional systems. The toolbox is exposed in the form of a Rust library with Python bindings.

## Setup

1. Download a recent `.whl` from [GitHub Releases](https://github.com/kylc/planning-kit/releases)
2. Run `pip install planning_kit_py<...>.whl` (replace `<...>` with the actual filename)
3. Test it: `python -c 'import planning_kit'`

## Usage

Plan around a sphere obstacle in $\mathbb{R}^3$ using the RRT-Connect planner:

https://github.com/kylc/planning-kit/blob/7579c0d1d3b0b75492f9c6b02b75627785d201b7/demo/01_intro.py#L1-L51

<p align="center">
    <img height="400" src="https://user-images.githubusercontent.com/233860/244265174-68069931-3bf9-410e-9c4f-5a0e3baff516.png">
</p>

## Details

### State Space

A state space (or configuration space) is defined by all possible configurations of the system. Arbitrary state space definitions are possible for use with the library.

For example, to define a Euclidean space (which has equivalent behavior to the built-in `StateSpace.euclidean`):

https://github.com/kylc/planning-kit/blob/7579c0d1d3b0b75492f9c6b02b75627785d201b7/demo/02_custom_space.py#L1-L32

<p align="center">
    <img height="400" src="https://user-images.githubusercontent.com/233860/244265175-285e9884-19be-418c-8c75-16cd02faaf0e.png">
</p>

### Constraints

Planning subject to manifold constraints is implemented via the unifying framework IMACS (implicit manifold configuration space) presented in [[KiMK19]](#KiMK19). Projection is used to adhere samples from an ambient space to the constrained manifold while preserving the planner properties of probabilistic completeness and asymptotic optimality.

In order to converge on an adhering sample, Newton's method is used to approximate the roots of the constraints. Constraint Jacobians (required for Newton's method) are automatically computed via central finite difference or complex-step differentiation, or are provided analytically.

For example, we can define a constraint which confines the state space to the surface of a unit sphere. With this constraint, we can define a state space which is automatically projected into the manifold. This state space can be used like any other, e.g. for use in motion planning algorithms.

https://github.com/kylc/planning-kit/blob/7579c0d1d3b0b75492f9c6b02b75627785d201b7/demo/03_constraints.py#L1-L33

<p align="center">
    <img height="400" src="https://user-images.githubusercontent.com/233860/244265172-db2e12b8-6100-41d0-bdbe-75b8b8af4b6b.png">
</p>

### Planning

| Planner     | Category     | Optimal | Reference           |
|-------------|--------------|---------|---------------------|
| PRM         | Multi-query  | :x:     | [[KSLO96]](#KSLO96) |
| RRT         | Single-query | :x:     | [[LaVa98]](#LaVa98) |
| RRT-Connect | Single-query | :x:     | [[KuLa00]](#KuLa00) |

### Post-processing

TODO

## Application

One potential application is to plan a joint-space trajectory for a robot arm which does not collide with obstacles in the environment. A predefined setup for this problem is defined in `planning_kit.KinematicChainProblem`. An example demonstrating its usage is provided:

https://github.com/kylc/planning-kit/blob/7579c0d1d3b0b75492f9c6b02b75627785d201b7/demo/04_arm_obstacle.py#L1-L68

https://user-images.githubusercontent.com/233860/244271528-99e51d06-ac2d-44eb-8861-74b1e960c59c.webm

## References

<a id="KiMK19">[KiMK19]:</a>
Z. Kingston, M. Moll, and L. E. Kavraki, "Exploring implicit spaces for constrained sampling-based planning," The International Journal of Robotics Research, vol. 38, no. 10–11, pp. 1151–1178, Sep. 2019, doi: 10.1177/0278364919868530.

<a id="KSLO96">[KSLO96]:</a>
L. E. Kavraki, P. Svestka, J.-C. Latombe, and M. H. Overmars, "Probabilistic roadmaps for path planning in high-dimensional configuration spaces," IEEE Transactions on Robotics and Automation, vol. 12, no. 4, pp. 566–580, Aug. 1996, doi: 10.1109/70.508439.

<a id="LaVa98">[LaVa98]:</a>
S. LaValle, "Rapidly-exploring random trees : a new tool for path planning," The annual research report, 1998, Accessed: May 26, 2023. [Online]. Available: https://www.semanticscholar.org/paper/Rapidly-exploring-random-trees-%3A-a-new-tool-for-LaValle/d967d9550f831a8b3f5cb00f8835a4c866da60ad

<a id="KuLa00">[KuLa00]:</a>
J. J. Kuffner and S. M. LaValle, "RRT-connect: An efficient approach to single-query path planning," in Proceedings 2000 ICRA. Millennium Conference. IEEE International Conference on Robotics and Automation. Symposia Proceedings (Cat. No.00CH37065), San Francisco, CA, USA: IEEE, 2000, pp. 995–1001. doi: 10.1109/ROBOT.2000.844730.
