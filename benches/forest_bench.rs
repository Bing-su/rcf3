use arcf::Forest;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use pprof::criterion::{Output, PProfProfiler};

fn build_forest(dim: usize, trees: usize, capacity: usize) -> Forest {
    Forest::builder(dim, 1)
        .num_trees(trees)
        .capacity(capacity)
        .seed(42)
        .build()
        .unwrap()
}

fn bench_update(c: &mut Criterion) {
    let mut group = c.benchmark_group("update");
    for &(dim, trees, cap, n) in &[(8, 50, 256, 20_000), (16, 100, 512, 20_000)] {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("d{dim}_t{trees}_c{cap}")),
            &(dim, trees, cap, n),
            |b, &(dim, trees, cap, n)| {
                b.iter(|| {
                    let mut f = build_forest(dim, trees, cap);
                    let p = vec![0.1_f32; dim];
                    for _ in 0..n {
                        f.update(&p).unwrap();
                    }
                });
            },
        );
    }
    group.finish();
}

fn bench_score(c: &mut Criterion) {
    let mut f = build_forest(8, 100, 512);
    let p = vec![0.2_f32; 8];
    for _ in 0..20_000 {
        f.update(&p).unwrap();
    }

    c.bench_function("score_ready", |b| {
        b.iter(|| {
            let _ = f.score(&p).unwrap();
        });
    });
}

fn bench_near_neighbors(c: &mut Criterion) {
    let mut f = build_forest(8, 100, 512);
    let p = vec![0.3_f32; 8];
    for _ in 0..20_000 {
        f.update(&p).unwrap();
    }

    c.bench_function("near_neighbors_top10_p50", |b| {
        b.iter(|| {
            let _ = f.near_neighbors(&p, 10, 50).unwrap();
        });
    });
}

criterion_group!(
    name = benches;
    config = Criterion::default().with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = bench_update, bench_score, bench_near_neighbors
);
criterion_main!(benches);
