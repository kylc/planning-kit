use criterion::{black_box, criterion_group, criterion_main, Criterion};
use nalgebra::{DVector, Vector3};
use planning_kit::state::{DynamicEuclideanSpace, EuclideanSpace, StateSpace};

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("sample_uniform");
    group.throughput(criterion::Throughput::Elements(1));

    let lo = Vector3::new(0.0, 0.0, 0.0);
    let hi = Vector3::new(1.0, 1.0, 1.0);
    let space = EuclideanSpace::<3>::new(lo, hi);

    group.bench_function("static", |b| {
        b.iter(|| {
            black_box(space.sample_uniform());
        })
    });

    let lo = DVector::from_row_slice(&[0.0, 0.0, 0.0]);
    let hi = DVector::from_row_slice(&[1.0, 1.0, 1.0]);
    let dyn_space = DynamicEuclideanSpace::new(lo, hi);

    group.bench_function("dynamic", |b| {
        b.iter(|| {
            black_box(dyn_space.sample_uniform());
        })
    });

    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
