use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rcf3::MStream;

#[derive(Clone, Copy)]
struct MStreamCase {
    numeric_dim: usize,
    categorical_dim: usize,
    rows: usize,
    buckets: usize,
    events: usize,
}

impl MStreamCase {
    fn label(self) -> String {
        format!(
            "n{}_c{}_r{}_b{}",
            self.numeric_dim, self.categorical_dim, self.rows, self.buckets
        )
    }
}

const SAME_TIMESTAMP_CASES: &[MStreamCase] = &[
    MStreamCase {
        numeric_dim: 8,
        categorical_dim: 4,
        rows: 2,
        buckets: 256,
        events: 20_000,
    },
    MStreamCase {
        numeric_dim: 16,
        categorical_dim: 8,
        rows: 4,
        buckets: 512,
        events: 10_000,
    },
    MStreamCase {
        numeric_dim: 32,
        categorical_dim: 0,
        rows: 4,
        buckets: 512,
        events: 10_000,
    },
];

const ADVANCING_TIMESTAMP_CASES: &[MStreamCase] = &[
    MStreamCase {
        numeric_dim: 8,
        categorical_dim: 4,
        rows: 2,
        buckets: 256,
        events: 10_000,
    },
    MStreamCase {
        numeric_dim: 16,
        categorical_dim: 8,
        rows: 4,
        buckets: 512,
        events: 5_000,
    },
    MStreamCase {
        numeric_dim: 32,
        categorical_dim: 0,
        rows: 4,
        buckets: 512,
        events: 5_000,
    },
];

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

fn build_mstream_for_case(case: MStreamCase) -> MStream {
    build_mstream(
        case.numeric_dim,
        case.categorical_dim,
        case.rows,
        case.buckets,
    )
}

fn run_same_timestamp_case(case: MStreamCase) {
    let mut detector = build_mstream_for_case(case);
    let numeric = vec![0.5_f32; case.numeric_dim];
    let categorical = vec![1_i64; case.categorical_dim];

    for _ in 0..case.events {
        let _ = detector
            .update_and_score(&numeric, &categorical, 1)
            .unwrap();
    }
}

fn run_advancing_timestamp_case(case: MStreamCase) {
    let mut detector = build_mstream_for_case(case);
    let numeric = vec![0.5_f32; case.numeric_dim];
    let categorical = vec![1_i64; case.categorical_dim];

    for ts in 1..=case.events as u64 {
        let _ = detector
            .update_and_score(&numeric, &categorical, ts)
            .unwrap();
    }
}

fn bench_update_and_score_cases(
    c: &mut Criterion,
    group_name: &str,
    cases: &[MStreamCase],
    run_case: fn(MStreamCase),
) {
    let mut group = c.benchmark_group(group_name);
    for &case in cases {
        group.throughput(Throughput::Elements(case.events as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(case.label()),
            &case,
            |b, &case| {
                b.iter(|| run_case(case));
            },
        );
    }
    group.finish();
}

fn bench_update_and_score_same_timestamp(c: &mut Criterion) {
    bench_update_and_score_cases(
        c,
        "mstream_update_and_score_same_ts",
        SAME_TIMESTAMP_CASES,
        run_same_timestamp_case,
    );
}

fn bench_update_and_score_advancing_timestamp(c: &mut Criterion) {
    bench_update_and_score_cases(
        c,
        "mstream_update_and_score_advancing_ts",
        ADVANCING_TIMESTAMP_CASES,
        run_advancing_timestamp_case,
    );
}

criterion_group!(
    name = benches;
    config = Criterion::default().measurement_time(Duration::from_secs(8));
    targets = bench_update_and_score_same_timestamp, bench_update_and_score_advancing_timestamp
);
criterion_main!(benches);
