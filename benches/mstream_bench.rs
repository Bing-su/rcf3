use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rcf3::MStream;

fn build_mstream(
    numeric_dim: usize,
    categorical_dim: usize,
    rows: usize,
    buckets: usize,
) -> MStream {
    MStream::builder(numeric_dim, categorical_dim)
        .seed(42)
        .alpha(0.8)
        .num_rows(rows)
        .num_buckets(buckets)
        .build()
        .unwrap()
}

fn bench_update_and_score_same_timestamp(c: &mut Criterion) {
    let mut group = c.benchmark_group("mstream_update_and_score_same_ts");
    for &(numeric_dim, categorical_dim, rows, buckets, n) in &[
        (8, 4, 2, 256, 20_000),
        (16, 8, 4, 512, 10_000),
        (32, 0, 4, 512, 10_000),
    ] {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!(
                "n{numeric_dim}_c{categorical_dim}_r{rows}_b{buckets}"
            )),
            &(numeric_dim, categorical_dim, rows, buckets, n),
            |b, &(numeric_dim, categorical_dim, rows, buckets, n)| {
                b.iter(|| {
                    let mut d = build_mstream(numeric_dim, categorical_dim, rows, buckets);
                    let numeric = vec![0.5_f32; numeric_dim];
                    let categorical = vec![1_i64; categorical_dim];
                    for _ in 0..n {
                        let _ = d.update_and_score(&numeric, &categorical, 1).unwrap();
                    }
                });
            },
        );
    }
    group.finish();
}

fn bench_update_and_score_advancing_timestamp(c: &mut Criterion) {
    let mut group = c.benchmark_group("mstream_update_and_score_advancing_ts");
    for &(numeric_dim, categorical_dim, rows, buckets, n) in &[
        (8, 4, 2, 256, 10_000),
        (16, 8, 4, 512, 5_000),
        (32, 0, 4, 512, 5_000),
    ] {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!(
                "n{numeric_dim}_c{categorical_dim}_r{rows}_b{buckets}"
            )),
            &(numeric_dim, categorical_dim, rows, buckets, n),
            |b, &(numeric_dim, categorical_dim, rows, buckets, n)| {
                b.iter(|| {
                    let mut d = build_mstream(numeric_dim, categorical_dim, rows, buckets);
                    let numeric = vec![0.5_f32; numeric_dim];
                    let categorical = vec![1_i64; categorical_dim];
                    for ts in 1..=n as u64 {
                        let _ = d.update_and_score(&numeric, &categorical, ts).unwrap();
                    }
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default().measurement_time(Duration::from_secs(8));
    targets = bench_update_and_score_same_timestamp, bench_update_and_score_advancing_timestamp
);
criterion_main!(benches);
