use std::time::Duration;

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rcf3::MStream;

#[derive(Clone, Copy)]
struct MStreamCase {
    numeric_dim: usize,
    categorical_dim: usize,
    rows: usize,
    buckets: usize,
    events: usize,
}

#[derive(Clone, Copy)]
enum TimestampMode {
    Same(u64),
    Advancing,
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

fn build_case_input(case: MStreamCase) -> (MStream, Vec<f32>, Vec<i64>) {
    let detector = build_mstream_for_case(case);
    let numeric = vec![0.5_f32; case.numeric_dim];
    let categorical = vec![1_i64; case.categorical_dim];
    (detector, numeric, categorical)
}

fn timestamp_for(mode: TimestampMode, offset: usize) -> u64 {
    match mode {
        TimestampMode::Same(ts) => ts,
        TimestampMode::Advancing => offset as u64 + 1,
    }
}

fn run_update_and_score_case(
    detector: &mut MStream,
    numeric: &[f32],
    categorical: &[i64],
    events: usize,
    timestamp_mode: TimestampMode,
) {
    for offset in 0..events {
        let ts = timestamp_for(timestamp_mode, offset);
        let _ = detector.update_and_score(numeric, categorical, ts).unwrap();
    }
}

fn bench_update_and_score_cases(
    c: &mut Criterion,
    group_name: &str,
    cases: &[MStreamCase],
    timestamp_mode: TimestampMode,
) {
    let mut group = c.benchmark_group(group_name);
    for &case in cases {
        group.throughput(Throughput::Elements(case.events as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(case.label()),
            &case,
            |b, &case| {
                b.iter_batched_ref(
                    || build_case_input(case),
                    |(detector, numeric, categorical)| {
                        run_update_and_score_case(
                            detector,
                            numeric,
                            categorical,
                            case.events,
                            timestamp_mode,
                        );
                    },
                    BatchSize::SmallInput,
                );
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
        TimestampMode::Same(1),
    );
}

fn bench_update_and_score_advancing_timestamp(c: &mut Criterion) {
    bench_update_and_score_cases(
        c,
        "mstream_update_and_score_advancing_ts",
        ADVANCING_TIMESTAMP_CASES,
        TimestampMode::Advancing,
    );
}

criterion_group!(
    name = benches;
    config = Criterion::default().measurement_time(Duration::from_secs(8));
    targets = bench_update_and_score_same_timestamp, bench_update_and_score_advancing_timestamp
);
criterion_main!(benches);
