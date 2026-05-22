use std::time::Duration;

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use ndarray::{Array2, ArrayView1, Axis};
use rcf3::{Forest, MStream, OnlineIForest};

const DIM: usize = 8;
const EVENTS: usize = 2_000;
const WARMUP_EVENTS: usize = 20_000;

#[derive(Clone, Copy)]
struct MStreamScenario {
    label: &'static str,
    categorical_dim: usize,
    rows: usize,
    buckets: usize,
}

struct StreamData {
    numeric: Array2<f32>,
    categorical: Array2<i64>,
}

impl StreamData {
    fn new(len: usize, categorical_dim: usize) -> Self {
        Self {
            numeric: numeric_stream(len),
            categorical: categorical_stream(len, categorical_dim),
        }
    }

    fn for_each_numeric_row(&self, mut visit: impl FnMut(&[f32])) {
        for point in self.numeric.axis_iter(Axis(0)) {
            visit(row_slice(&point));
        }
    }

    fn for_each_mstream_row(&self, mut visit: impl FnMut(usize, &[f32], &[i64])) {
        for (offset, (numeric, categorical)) in self
            .numeric
            .axis_iter(Axis(0))
            .zip(self.categorical.axis_iter(Axis(0)))
            .enumerate()
        {
            visit(
                offset,
                row_slice(&numeric),
                categorical_row_slice(&categorical),
            );
        }
    }
}

fn numeric_stream(len: usize) -> Array2<f32> {
    Array2::from_shape_fn((len, DIM), |(event_idx, feature_idx)| {
        let x = event_idx as f32 * 0.01 + feature_idx as f32 * 0.1;
        2.0 + x.sin() + 0.25 * (x * 0.5).cos()
    })
}

fn categorical_stream(len: usize, categorical_dim: usize) -> Array2<i64> {
    Array2::from_shape_fn((len, categorical_dim), |(event_idx, feature_idx)| {
        ((event_idx / (feature_idx + 1)) % 17) as i64
    })
}

fn row_slice<'a>(row: &'a ArrayView1<'a, f32>) -> &'a [f32] {
    row.as_slice()
        .expect("numeric_stream rows should be contiguous")
}

fn categorical_row_slice<'a>(row: &'a ArrayView1<'a, i64>) -> &'a [i64] {
    row.as_slice()
        .expect("categorical_stream rows should be contiguous")
}

fn build_ready_forest(warmup: &StreamData) -> Forest {
    let mut forest = Forest::builder(DIM)
        .shingle_size(1)
        .num_trees(50)
        .capacity(256)
        .seed(42)
        .build()
        .unwrap();

    warmup.for_each_numeric_row(|point| forest.update(point).unwrap());

    forest
}

fn build_ready_onlineiforest(warmup: &StreamData) -> OnlineIForest {
    let mut detector = OnlineIForest::builder(DIM)
        .num_trees(50)
        .window_size(256)
        .max_leaf_samples(8)
        .seed(42)
        .build()
        .unwrap();

    warmup.for_each_numeric_row(|point| detector.update(point).unwrap());

    detector
}

fn build_ready_mstream(warmup: &StreamData, scenario: MStreamScenario) -> MStream {
    let mut detector = MStream::builder(DIM, scenario.categorical_dim)
        .seed(42)
        .alpha(0.8)
        .num_rows(scenario.rows)
        .num_buckets(scenario.buckets)
        .build()
        .unwrap();

    warmup.for_each_mstream_row(|offset, numeric, categorical| {
        detector
            .update_and_score(numeric, categorical, (offset + 1) as u64)
            .unwrap();
    });

    detector
}

fn run_forest_score_then_update(detector: &mut Forest, events: &StreamData) {
    events.for_each_numeric_row(|point| {
        let _ = detector.score(point).unwrap();
        detector.update(point).unwrap();
    });
}

fn run_onlineiforest_update_and_score(detector: &mut OnlineIForest, events: &StreamData) {
    events.for_each_numeric_row(|point| {
        let _ = detector.update_and_score(point).unwrap();
    });
}

fn run_onlineiforest_score_then_update(detector: &mut OnlineIForest, events: &StreamData) {
    events.for_each_numeric_row(|point| {
        let _ = detector.score(point).unwrap();
        detector.update(point).unwrap();
    });
}

fn run_mstream_update_and_score(detector: &mut MStream, events: &StreamData) {
    let start_ts = detector.current_time().unwrap_or(0);
    events.for_each_mstream_row(|offset, numeric, categorical| {
        let _ = detector
            .update_and_score(numeric, categorical, start_ts + offset as u64 + 1)
            .unwrap();
    });
}

fn bench_numeric_stream_step(c: &mut Criterion) {
    let warmup = StreamData::new(WARMUP_EVENTS, 0);
    let events = StreamData::new(EVENTS, 0);
    let forest = build_ready_forest(&warmup);
    let onlineiforest = build_ready_onlineiforest(&warmup);

    let mut group = c.benchmark_group("numeric_stream_step");
    group.throughput(Throughput::Elements(EVENTS as u64));

    group.bench_with_input(
        BenchmarkId::new("forest_score_then_update", "d8_t50_c256"),
        &events,
        |b, events| {
            b.iter_batched(
                || forest.clone(),
                |mut detector| run_forest_score_then_update(&mut detector, events),
                BatchSize::SmallInput,
            );
        },
    );

    group.bench_with_input(
        BenchmarkId::new("onlineiforest_update_and_score", "d8_t50_w256_l8"),
        &events,
        |b, events| {
            b.iter_batched(
                || onlineiforest.clone(),
                |mut detector| run_onlineiforest_update_and_score(&mut detector, events),
                BatchSize::SmallInput,
            );
        },
    );

    group.bench_with_input(
        BenchmarkId::new("onlineiforest_score_then_update", "d8_t50_w256_l8"),
        &events,
        |b, events| {
            b.iter_batched(
                || onlineiforest.clone(),
                |mut detector| run_onlineiforest_score_then_update(&mut detector, events),
                BatchSize::SmallInput,
            );
        },
    );

    group.finish();
}

fn bench_mstream_stream_step(c: &mut Criterion) {
    let scenarios = [
        MStreamScenario {
            label: "n8_c0_r2_b256",
            categorical_dim: 0,
            rows: 2,
            buckets: 256,
        },
        MStreamScenario {
            label: "n8_c4_r2_b256",
            categorical_dim: 4,
            rows: 2,
            buckets: 256,
        },
    ];

    let mut group = c.benchmark_group("mstream_stream_step");
    group.throughput(Throughput::Elements(EVENTS as u64));

    for scenario in scenarios {
        let warmup = StreamData::new(WARMUP_EVENTS, scenario.categorical_dim);
        let events = StreamData::new(EVENTS, scenario.categorical_dim);
        let mstream = build_ready_mstream(&warmup, scenario);

        group.bench_with_input(
            BenchmarkId::new("update_and_score", scenario.label),
            &events,
            |b, events| {
                b.iter_batched(
                    || mstream.clone(),
                    |mut detector| run_mstream_update_and_score(&mut detector, events),
                    BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default().measurement_time(Duration::from_secs(10));
    targets = bench_numeric_stream_step, bench_mstream_stream_step
);
criterion_main!(benches);
