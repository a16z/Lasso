use ark_bn254::Fr;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use liblasso::jolt::subtable::eq::EqSubtable;
use liblasso::jolt::subtable::eq_abs::EqAbsSubtable;
use liblasso::jolt::subtable::LassoSubtable;

fn materialize_subtable(c: &mut Criterion) {
    let mut group = c.benchmark_group("DenseMLPoly Evaluation");

    group.bench_function("Materialize EQ subtable", |b| {
        b.iter(|| {
            let table = EqSubtable::<Fr>::new();
            black_box(table.materialize(1 << 16));
        })
    });

    group.bench_function("Materialize EQAbs subtable", |b| {
        b.iter(|| {
            let table = EqAbsSubtable::<Fr>::new();
            black_box(table.materialize(1 << 16));
        })
    });

    group.finish();
}

criterion_group!(benches, materialize_subtable);
criterion_main!(benches);
