use std::time::Duration;

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use ndarray::{Array2, ArrayView1, Axis};
use rcf3::{FeatureSketch, Forest, MStream, OnlineIForest};

const DIM: usize = 8;
const EVENTS: usize = 2_000;
const WARMUP_EVENTS: usize = 20_000;

#[derive(Clone, Copy)]
enum ConfigSize {
    Small,
    Default,
}

#[derive(Clone, Copy)]
struct ForestScenario {
    label: &'static str,
    size: ConfigSize,
}

#[derive(Clone, Copy)]
struct OnlineIForestScenario {
    label: &'static str,
    size: ConfigSize,
}

#[derive(Clone, Copy)]
struct MStreamScenario {
    label: &'static str,
    size: ConfigSize,
    categorical_dim: usize,
}

#[derive(Clone, Copy)]
struct FeatureSketchScenario {
    label: &'static str,
    size: ConfigSize,
    value_projection_dims: usize,
    presence_projection_dims: usize,
    chains_per_ensemble: usize,
    chain_depth: usize,
    sketch_rows: usize,
    sketch_buckets: usize,
    decay_half_life: u64,
}

struct StreamData {
    numeric: Array2<f32>,
    numeric_f64: Array2<f64>,
    categorical: Array2<i64>,
}

struct FeatureStreamData {
    events: Vec<Vec<(String, f64)>>,
}

impl StreamData {
    fn new(len: usize, categorical_dim: usize) -> Self {
        Self {
            numeric: numeric_stream(len),
            numeric_f64: numeric_stream_f64(len),
            categorical: categorical_stream(len, categorical_dim),
        }
    }

    fn for_each_numeric_row(&self, mut visit: impl FnMut(&[f32])) {
        for point in self.numeric.axis_iter(Axis(0)) {
            visit(row_slice(&point));
        }
    }

    fn for_each_mstream_row(&self, mut visit: impl FnMut(usize, &[f64], &[i64])) {
        for (offset, (numeric, categorical)) in self
            .numeric_f64
            .axis_iter(Axis(0))
            .zip(self.categorical.axis_iter(Axis(0)))
            .enumerate()
        {
            visit(
                offset,
                row_slice_f64(&numeric),
                categorical_row_slice(&categorical),
            );
        }
    }
}

impl FeatureStreamData {
    fn new(len: usize) -> Self {
        Self {
            events: (0..len).map(feature_event).collect(),
        }
    }

    fn for_each_event(&self, mut visit: impl FnMut(&[(String, f64)])) {
        for event in &self.events {
            visit(event);
        }
    }
}

fn numeric_stream(len: usize) -> Array2<f32> {
    Array2::from_shape_fn((len, DIM), |(event_idx, feature_idx)| {
        let x = event_idx as f32 * 0.01 + feature_idx as f32 * 0.1;
        2.0 + x.sin() + 0.25 * (x * 0.5).cos()
    })
}

fn numeric_stream_f64(len: usize) -> Array2<f64> {
    Array2::from_shape_fn((len, DIM), |(event_idx, feature_idx)| {
        let x = event_idx as f64 * 0.01 + feature_idx as f64 * 0.1;
        2.0 + x.sin() + 0.25 * (x * 0.5).cos()
    })
}

fn categorical_stream(len: usize, categorical_dim: usize) -> Array2<i64> {
    Array2::from_shape_fn((len, categorical_dim), |(event_idx, feature_idx)| {
        ((event_idx / (feature_idx + 1)) % 17) as i64
    })
}

fn feature_event(event_idx: usize) -> Vec<(String, f64)> {
    let endpoint = match event_idx % 4 {
        0 => "login",
        1 => "checkout",
        2 => "account",
        _ => "search",
    };
    let method = if event_idx.is_multiple_of(5) {
        "POST"
    } else {
        "GET"
    };
    let status = if event_idx.is_multiple_of(97) {
        500
    } else {
        200
    };
    let latency = 45.0 + ((event_idx as f64) * 0.013).sin() * 8.0;
    let bytes_out = 2.0 + ((event_idx as f64) * 0.007).cos();

    let mut features = vec![
        (format!("endpoint:{endpoint}"), 1.0),
        (format!("method:{method}"), 1.0),
        (format!("status:{status}"), 1.0),
        ("service:payments-api".to_string(), 1.0),
        ("auth:authenticated".to_string(), 1.0),
        ("device:known".to_string(), 1.0),
        ("mfa:present".to_string(), 1.0),
        ("latency_ms".to_string(), latency),
        ("bytes_out_kib".to_string(), bytes_out),
        ("failed_auths".to_string(), 0.0),
    ];

    if event_idx.is_multiple_of(37) {
        features.push((format!("partner:{}", event_idx % 11), 1.0));
    }
    if event_idx.is_multiple_of(211) {
        features.push(("header:x-forwarded-admin".to_string(), 1.0));
    }

    features
}

fn row_slice<'a>(row: &'a ArrayView1<'a, f32>) -> &'a [f32] {
    row.as_slice()
        .expect("numeric_stream rows should be contiguous")
}

fn row_slice_f64<'a>(row: &'a ArrayView1<'a, f64>) -> &'a [f64] {
    row.as_slice()
        .expect("numeric_stream_f64 rows should be contiguous")
}

fn categorical_row_slice<'a>(row: &'a ArrayView1<'a, i64>) -> &'a [i64] {
    row.as_slice()
        .expect("categorical_stream rows should be contiguous")
}

fn build_ready_forest(warmup: &StreamData, scenario: ForestScenario) -> Forest {
    let mut builder = Forest::builder(DIM).seed(42);
    if matches!(scenario.size, ConfigSize::Small) {
        builder = builder.num_trees(16).capacity(128);
    }
    let mut forest = builder.build().unwrap();

    warmup.for_each_numeric_row(|point| forest.update(point).unwrap());

    forest
}

fn build_ready_onlineiforest(
    warmup: &StreamData,
    scenario: OnlineIForestScenario,
) -> OnlineIForest {
    let mut builder = OnlineIForest::builder(DIM).seed(42);
    if matches!(scenario.size, ConfigSize::Small) {
        builder = builder.num_trees(16).window_size(512).max_leaf_samples(8);
    }
    let mut detector = builder.build().unwrap();

    warmup.for_each_numeric_row(|point| detector.update(point).unwrap());

    detector
}

fn build_ready_featuresketch(
    warmup: &FeatureStreamData,
    scenario: FeatureSketchScenario,
) -> FeatureSketch {
    let mut builder = FeatureSketch::builder().seed(42);
    if matches!(scenario.size, ConfigSize::Small) {
        builder = builder
            .value_projection_dims(scenario.value_projection_dims)
            .presence_projection_dims(scenario.presence_projection_dims)
            .chains_per_ensemble(scenario.chains_per_ensemble)
            .chain_depth(scenario.chain_depth)
            .sketch_rows(scenario.sketch_rows)
            .sketch_buckets(scenario.sketch_buckets)
            .decay_half_life(scenario.decay_half_life);
    }
    let mut detector = builder.build().unwrap();

    warmup.for_each_event(|event| {
        detector
            .update(event.iter().map(|(name, value)| (name.as_str(), *value)))
            .unwrap();
    });

    detector
}

fn build_ready_mstream(warmup: &StreamData, scenario: MStreamScenario) -> MStream {
    let mut builder = MStream::builder(DIM, scenario.categorical_dim).seed(42);
    if matches!(scenario.size, ConfigSize::Small) {
        builder = builder.num_rows(2).num_buckets(256);
    }
    let mut detector = builder.build().unwrap();

    warmup.for_each_mstream_row(|offset, numeric, categorical| {
        detector
            .update_and_score(numeric, categorical, (offset + 1) as u64)
            .unwrap();
    });

    detector
}

fn run_forest_update_and_score(detector: &mut Forest, events: &StreamData) {
    events.for_each_numeric_row(|point| {
        let _ = detector.update_and_score(point).unwrap();
    });
}

fn run_onlineiforest_update_and_score(detector: &mut OnlineIForest, events: &StreamData) {
    events.for_each_numeric_row(|point| {
        let _ = detector.update_and_score(point).unwrap();
    });
}

fn run_featuresketch_update_and_score(detector: &mut FeatureSketch, events: &FeatureStreamData) {
    events.for_each_event(|event| {
        let _ = detector
            .update_and_score(event.iter().map(|(name, value)| (name.as_str(), *value)))
            .unwrap();
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
    let forest_scenarios = [
        ForestScenario {
            label: "small_d8_t16_c128",
            size: ConfigSize::Small,
        },
        ForestScenario {
            label: "default_d8_t50_c256",
            size: ConfigSize::Default,
        },
    ];
    let onlineiforest_scenarios = [
        OnlineIForestScenario {
            label: "small_d8_t16_w512_l8",
            size: ConfigSize::Small,
        },
        OnlineIForestScenario {
            label: "default_d8_t32_w2048_l32",
            size: ConfigSize::Default,
        },
    ];

    let warmup = StreamData::new(WARMUP_EVENTS, 0);
    let events = StreamData::new(EVENTS, 0);

    let mut group = c.benchmark_group("numeric_stream_step");
    group.throughput(Throughput::Elements(EVENTS as u64));

    for scenario in forest_scenarios {
        let forest = build_ready_forest(&warmup, scenario);
        group.bench_with_input(
            BenchmarkId::new("forest_update_and_score", scenario.label),
            &events,
            |b, events| {
                b.iter_batched(
                    || forest.clone(),
                    |mut detector| run_forest_update_and_score(&mut detector, events),
                    BatchSize::SmallInput,
                );
            },
        );
    }

    for scenario in onlineiforest_scenarios {
        let onlineiforest = build_ready_onlineiforest(&warmup, scenario);
        group.bench_with_input(
            BenchmarkId::new("onlineiforest_update_and_score", scenario.label),
            &events,
            |b, events| {
                b.iter_batched(
                    || onlineiforest.clone(),
                    |mut detector| run_onlineiforest_update_and_score(&mut detector, events),
                    BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

fn bench_mstream_stream_step(c: &mut Criterion) {
    let scenarios = [
        MStreamScenario {
            label: "small_n8_c0_r2_b256",
            size: ConfigSize::Small,
            categorical_dim: 0,
        },
        MStreamScenario {
            label: "default_n8_c0_r2_b1024",
            size: ConfigSize::Default,
            categorical_dim: 0,
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

fn bench_featuresketch_stream_step(c: &mut Criterion) {
    let scenarios = [
        FeatureSketchScenario {
            label: "small_api_sparse_v16_p16_c12_d6_r2_b512",
            size: ConfigSize::Small,
            value_projection_dims: 16,
            presence_projection_dims: 16,
            chains_per_ensemble: 12,
            chain_depth: 6,
            sketch_rows: 2,
            sketch_buckets: 512,
            decay_half_life: 512,
        },
        FeatureSketchScenario {
            label: "default_api_sparse_v32_p32_c16_d8_r2_b2048",
            size: ConfigSize::Default,
            value_projection_dims: 32,
            presence_projection_dims: 32,
            chains_per_ensemble: 16,
            chain_depth: 8,
            sketch_rows: 2,
            sketch_buckets: 2048,
            decay_half_life: 2048,
        },
    ];

    let warmup = FeatureStreamData::new(WARMUP_EVENTS);
    let events = FeatureStreamData::new(EVENTS);

    let mut group = c.benchmark_group("featuresketch_stream_step");
    group.throughput(Throughput::Elements(EVENTS as u64));

    for scenario in scenarios {
        let featuresketch = build_ready_featuresketch(&warmup, scenario);

        group.bench_with_input(
            BenchmarkId::new("update_and_score", scenario.label),
            &events,
            |b, events| {
                b.iter_batched(
                    || featuresketch.clone(),
                    |mut detector| run_featuresketch_update_and_score(&mut detector, events),
                    BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default().measurement_time(Duration::from_secs(15));
    targets = bench_numeric_stream_step, bench_mstream_stream_step, bench_featuresketch_stream_step
);
criterion_main!(benches);
