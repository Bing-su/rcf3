use std::fmt;
use std::hint::black_box;
use std::time::Duration;

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rcf3::FeatureSketch;

const WARMUP_EVENTS: usize = 20_000;

#[derive(Clone, Copy)]
struct FeatureSketchCase {
    name: &'static str,
    feature_count: usize,
    unique_feature_names: usize,
    events: usize,
}

impl fmt::Display for FeatureSketchCase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.unique_feature_names == self.feature_count {
            write!(f, "{}_f{}", self.name, self.feature_count)
        } else {
            write!(
                f,
                "{}_f{}_u{}",
                self.name, self.feature_count, self.unique_feature_names
            )
        }
    }
}

const STREAM_CASES: &[FeatureSketchCase] = &[
    FeatureSketchCase {
        name: "sparse",
        feature_count: 8,
        unique_feature_names: 8,
        events: 10_000,
    },
    FeatureSketchCase {
        name: "default_like",
        feature_count: 16,
        unique_feature_names: 16,
        events: 5_000,
    },
    FeatureSketchCase {
        name: "wide",
        feature_count: 64,
        unique_feature_names: 64,
        events: 2_000,
    },
    FeatureSketchCase {
        name: "duplicate_heavy",
        feature_count: 64,
        unique_feature_names: 8,
        events: 2_000,
    },
];

struct FeatureEvents {
    events: Vec<Vec<(String, f64)>>,
}

impl FeatureEvents {
    fn new(len: usize, case: FeatureSketchCase) -> Self {
        Self {
            events: (0..len)
                .map(|event_idx| feature_event(event_idx, case))
                .collect(),
        }
    }

    fn for_each_event(&self, mut visit: impl FnMut(&[(String, f64)])) {
        for event in &self.events {
            visit(event);
        }
    }
}

fn feature_event(event_idx: usize, case: FeatureSketchCase) -> Vec<(String, f64)> {
    let mut features = Vec::with_capacity(case.feature_count);
    for feature_idx in 0..case.feature_count {
        let feature_name_idx = feature_idx % case.unique_feature_names;
        let feature_name = format!("feature:{feature_name_idx}:bucket:{}", event_idx % 31);
        let phase = event_idx as f64 * 0.013 + feature_idx as f64 * 0.17;
        let value = if feature_idx % 5 == 0 {
            1.0
        } else {
            phase.sin() * 10.0
        };
        features.push((feature_name, value));
    }
    features
}

fn feature_items(event: &[(String, f64)]) -> impl Iterator<Item = (&str, f64)> {
    event.iter().map(|(name, value)| (name.as_str(), *value))
}

fn build_featuresketch() -> FeatureSketch {
    FeatureSketch::builder().seed(42).build().unwrap()
}

fn build_ready_featuresketch(warmup: &FeatureEvents) -> FeatureSketch {
    let mut detector = build_featuresketch();
    warmup.for_each_event(|event| {
        detector.update(feature_items(event)).unwrap();
    });
    detector
}

fn score_events(detector: &FeatureSketch, events: &FeatureEvents) {
    events.for_each_event(|event| {
        let score = detector.score(feature_items(event)).unwrap();
        black_box(score);
    });
}

fn update_events(detector: &mut FeatureSketch, events: &FeatureEvents) {
    events.for_each_event(|event| {
        detector.update(feature_items(event)).unwrap();
    });
}

fn update_and_score_events(detector: &mut FeatureSketch, events: &FeatureEvents) {
    events.for_each_event(|event| {
        let score = detector.update_and_score(feature_items(event)).unwrap();
        black_box(score);
    });
}

fn score_then_update_events(detector: &mut FeatureSketch, events: &FeatureEvents) {
    events.for_each_event(|event| {
        let score = detector.score(feature_items(event)).unwrap();
        black_box(score);
        detector.update(feature_items(event)).unwrap();
    });
}

fn bench_operation(
    c: &mut Criterion,
    group_name: &str,
    mut run: impl FnMut(&mut FeatureSketch, &FeatureEvents) + Copy + 'static,
) {
    let mut group = c.benchmark_group(group_name);
    for &case in STREAM_CASES {
        let warmup = FeatureEvents::new(WARMUP_EVENTS, case);
        let events = FeatureEvents::new(case.events, case);
        let detector = build_ready_featuresketch(&warmup);

        group.throughput(Throughput::Elements(case.events as u64));
        group.bench_with_input(BenchmarkId::from_parameter(case), &events, |b, events| {
            b.iter_batched(
                || detector.clone(),
                |mut detector| run(&mut detector, events),
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

fn bench_score(c: &mut Criterion) {
    bench_operation(c, "featuresketch_score_ready", |detector, events| {
        score_events(detector, events);
    });
}

fn bench_update(c: &mut Criterion) {
    bench_operation(c, "featuresketch_update_ready", |detector, events| {
        update_events(detector, events);
    });
}

fn bench_update_and_score(c: &mut Criterion) {
    bench_operation(
        c,
        "featuresketch_update_and_score_ready",
        |detector, events| {
            update_and_score_events(detector, events);
        },
    );
}

fn bench_score_then_update(c: &mut Criterion) {
    bench_operation(
        c,
        "featuresketch_score_then_update_ready",
        |detector, events| {
            score_then_update_events(detector, events);
        },
    );
}

criterion_group!(
    name = benches;
    config = Criterion::default().measurement_time(Duration::from_secs(8));
    targets = bench_score, bench_update, bench_update_and_score, bench_score_then_update
);
criterion_main!(benches);
