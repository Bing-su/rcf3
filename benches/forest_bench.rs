use std::time::Duration;

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rcf3::Forest;

fn build_forest(dim: usize, trees: usize, capacity: usize) -> Forest {
    Forest::builder(dim)
        .shingle_size(1)
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

    group.throughput(Throughput::Elements(20_000));
    group.bench_function("steady_rejection_d8_t50_c256", |b| {
        b.iter_batched(
            || {
                let mut f = Forest::builder(8)
                    .shingle_size(1)
                    .num_trees(50)
                    .capacity(256)
                    .time_decay(f64::MIN_POSITIVE)
                    .seed(42)
                    .build()
                    .unwrap();
                let p = vec![0.1_f32; 8];
                for _ in 0..30_000 {
                    f.update(&p).unwrap();
                }
                (f, p)
            },
            |(mut f, p)| {
                for _ in 0..20_000 {
                    f.update(&p).unwrap();
                }
            },
            BatchSize::SmallInput,
        );
    });

    group.throughput(Throughput::Elements(500));
    group.bench_function("mixed_unique_d8_t50_c256", |b| {
        b.iter_batched(
            || build_forest(8, 50, 256),
            |mut f| {
                for i in 0..500 {
                    let base = i as f32 * 0.001;
                    let p = [
                        base.sin(),
                        base.cos(),
                        (base * 0.5).sin(),
                        (base * 0.5).cos(),
                        (base * 1.7).sin(),
                        (base * 1.7).cos(),
                        (base * 2.3).sin(),
                        (base * 2.3).cos(),
                    ];
                    f.update(&p).unwrap();
                }
            },
            BatchSize::SmallInput,
        );
    });
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

    let mut f_large = build_forest(8, 1024, 512);
    for _ in 0..20_000 {
        f_large.update(&p).unwrap();
    }

    c.bench_function("score_ready_t1024", |b| {
        b.iter(|| {
            let _ = f_large.score(&p).unwrap();
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

    let mut f_large = build_forest(8, 1024, 512);
    for _ in 0..20_000 {
        f_large.update(&p).unwrap();
    }

    c.bench_function("near_neighbors_top10_p50_t1024", |b| {
        b.iter(|| {
            let _ = f_large.near_neighbors(&p, 10, 50).unwrap();
        });
    });
}

fn bench_impute(c: &mut Criterion) {
    // Build a shingled forest (shingle_size=4, input_dim=2) so impute is meaningful.
    let mut f = Forest::builder(2)
        .shingle_size(4)
        .num_trees(100)
        .capacity(512)
        .seed(99)
        .build()
        .unwrap();
    for i in 0..20_000 {
        let obs = vec![(i as f32) * 0.001, (i as f32) * 0.002];
        f.update(&obs).unwrap();
    }
    // Query has full dim = 8; mask out last 2 dims.
    let query = vec![0.0f32; 8];
    let missing = vec![6, 7];

    c.bench_function("impute_2missing_of_8", |b| {
        b.iter(|| {
            let _ = f.impute(&query, &missing, 1.0).unwrap();
        });
    });
}

criterion_group!(
    name = benches;
    config = Criterion::default().measurement_time(Duration::from_secs(10));
    targets = bench_update, bench_score, bench_near_neighbors, bench_impute
);
criterion_main!(benches);
