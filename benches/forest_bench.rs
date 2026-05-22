use std::time::Duration;

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rcf3::Forest;

const UPDATE_EVENTS: usize = 20_000;
const STEADY_WARMUP_EVENTS: usize = 30_000;
const MIXED_UNIQUE_EVENTS: usize = 500;
const READY_FOREST_EVENTS: usize = UPDATE_EVENTS;

#[derive(Clone, Copy)]
struct ForestUpdateCase {
    dim: usize,
    trees: usize,
    capacity: usize,
    events: usize,
}

impl ForestUpdateCase {
    fn label(self) -> String {
        format!("d{}_t{}_c{}", self.dim, self.trees, self.capacity)
    }
}

const UPDATE_CASES: &[ForestUpdateCase] = &[
    ForestUpdateCase {
        dim: 8,
        trees: 50,
        capacity: 256,
        events: UPDATE_EVENTS,
    },
    ForestUpdateCase {
        dim: 16,
        trees: 100,
        capacity: 512,
        events: UPDATE_EVENTS,
    },
];

fn build_forest(dim: usize, trees: usize, capacity: usize) -> Forest {
    Forest::builder(dim)
        .shingle_size(1)
        .num_trees(trees)
        .capacity(capacity)
        .seed(42)
        .build()
        .unwrap()
}

fn build_steady_rejection_forest() -> Forest {
    Forest::builder(8)
        .shingle_size(1)
        .num_trees(50)
        .capacity(256)
        .time_decay(f64::MIN_POSITIVE)
        .seed(42)
        .build()
        .unwrap()
}

fn mixed_unique_point(i: usize) -> [f32; 8] {
    let base = i as f32 * 0.001;
    [
        base.sin(),
        base.cos(),
        (base * 0.5).sin(),
        (base * 0.5).cos(),
        (base * 1.7).sin(),
        (base * 1.7).cos(),
        (base * 2.3).sin(),
        (base * 2.3).cos(),
    ]
}

fn update_repeatedly(forest: &mut Forest, point: &[f32], events: usize) {
    for _ in 0..events {
        forest.update(point).unwrap();
    }
}

fn build_ready_forest(
    dim: usize,
    trees: usize,
    capacity: usize,
    point_value: f32,
) -> (Forest, Vec<f32>) {
    let mut forest = build_forest(dim, trees, capacity);
    let point = vec![point_value; dim];
    update_repeatedly(&mut forest, &point, READY_FOREST_EVENTS);
    (forest, point)
}

fn build_steady_rejection_input() -> (Forest, Vec<f32>) {
    let mut forest = build_steady_rejection_forest();
    let point = vec![0.1_f32; 8];
    update_repeatedly(&mut forest, &point, STEADY_WARMUP_EVENTS);
    (forest, point)
}

fn build_update_input(case: ForestUpdateCase) -> (Forest, Vec<f32>) {
    let forest = build_forest(case.dim, case.trees, case.capacity);
    let point = vec![0.1_f32; case.dim];
    (forest, point)
}

fn mixed_unique_points() -> Vec<[f32; 8]> {
    (0..MIXED_UNIQUE_EVENTS).map(mixed_unique_point).collect()
}

fn bench_update(c: &mut Criterion) {
    let mut group = c.benchmark_group("update");
    for &case in UPDATE_CASES {
        group.throughput(Throughput::Elements(case.events as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(case.label()),
            &case,
            |b, &case| {
                b.iter_batched_ref(
                    || build_update_input(case),
                    |(forest, point)| {
                        update_repeatedly(forest, point, case.events);
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }

    group.throughput(Throughput::Elements(UPDATE_EVENTS as u64));
    group.bench_function("steady_rejection_d8_t50_c256", |b| {
        b.iter_batched_ref(
            build_steady_rejection_input,
            |(forest, point)| {
                update_repeatedly(forest, point, UPDATE_EVENTS);
            },
            BatchSize::SmallInput,
        );
    });

    group.throughput(Throughput::Elements(MIXED_UNIQUE_EVENTS as u64));
    let points = mixed_unique_points();
    group.bench_function("mixed_unique_d8_t50_c256", |b| {
        b.iter_batched_ref(
            || build_forest(8, 50, 256),
            |forest| {
                for point in &points {
                    forest.update(point).unwrap();
                }
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

fn bench_score(c: &mut Criterion) {
    let (forest, point) = build_ready_forest(8, 100, 512, 0.2);

    c.bench_function("score_ready", |b| {
        b.iter(|| {
            let _ = forest.score(&point).unwrap();
        });
    });

    let (large_forest, large_point) = build_ready_forest(8, 1024, 512, 0.2);

    c.bench_function("score_ready_t1024", |b| {
        b.iter(|| {
            let _ = large_forest.score(&large_point).unwrap();
        });
    });
}

fn bench_near_neighbors(c: &mut Criterion) {
    let (forest, point) = build_ready_forest(8, 100, 512, 0.3);

    c.bench_function("near_neighbors_top10_p50", |b| {
        b.iter(|| {
            let _ = forest.near_neighbors(&point, 10, 50).unwrap();
        });
    });

    let (large_forest, large_point) = build_ready_forest(8, 1024, 512, 0.3);

    c.bench_function("near_neighbors_top10_p50_t1024", |b| {
        b.iter(|| {
            let _ = large_forest.near_neighbors(&large_point, 10, 50).unwrap();
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
