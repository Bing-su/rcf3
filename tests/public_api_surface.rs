/// Smoke test for the crate-level public facade exports.
use proptest::prelude::*;
use rcf3::{
    Attribution, FeatureSketch, FeatureSketchBuilder, FeatureSketchConfig, Forest, ForestBuilder,
    MStream, MStreamConfig, NeighborResult, OnlineIForest, OnlineIForestConfig, RcfConfig,
    RcfError,
};
#[cfg(feature = "serde")]
use rcf3::{MStreamBuilder, OnlineIForestBuilder};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    #[test]
    fn top_level_facade_exports_expected_user_facing_types(
        forest_seed in any::<u64>(),
        featuresketch_seed in any::<u64>(),
    ) {
    let config = RcfConfig::builder(2)
        .shingle_size(3)
        .capacity(64)
        .num_trees(7)
        .time_decay(0.01)
        .output_after(5)
        .internal_shingling(false)
        .initial_accept_fraction(0.25)
        .build();

    assert_eq!(config.input_dim(), 2);
    assert_eq!(config.shingle_size(), 3);
    assert_eq!(config.capacity(), 64);
    assert_eq!(config.num_trees(), 7);
    assert_eq!(config.time_decay(), 0.01);
    assert_eq!(config.output_after(), 5);
    assert!(!config.internal_shingling());
    assert_eq!(config.initial_accept_fraction(), 0.25);

    let builder: ForestBuilder = Forest::builder(2);
    let forest = builder.seed(forest_seed).build().unwrap();
    prop_assert_eq!(forest.config().input_dim(), 2);

    let attr = Attribution {
        below: 1.25,
        above: 0.75,
    };
    prop_assert_eq!(attr.total(), 2.0);

    let neighbor = NeighborResult {
        score: 0.5,
        point: vec![1.0, 2.0],
        distance: 3.0,
    };
    prop_assert_eq!(neighbor.point, vec![1.0, 2.0]);

    let feature_config = FeatureSketchConfig::builder()
        .value_projection_dims(4)
        .presence_projection_dims(4)
        .chains_per_ensemble(2)
        .chain_depth(2)
        .sketch_buckets(32)
        .build();
    prop_assert_eq!(feature_config.value_projection_dims(), 4);

    let builder: FeatureSketchBuilder<_> = FeatureSketch::builder()
        .value_projection_dims(4)
        .presence_projection_dims(4)
        .chains_per_ensemble(2)
        .chain_depth(2)
        .sketch_buckets(32)
        .seed(featuresketch_seed);
    let mut detector = builder.build().unwrap();
    detector.update([("feature", 1.0)]).unwrap();
    prop_assert!(detector.score([("feature", 1.0)]).unwrap().is_finite());
    }
}

#[cfg(feature = "serde")]
#[test]
fn builders_keep_config_defaults_when_optional_values_are_omitted() {
    let forest: ForestBuilder = ForestBuilder::new(2);
    let online: OnlineIForestBuilder = OnlineIForest::builder(2);
    let mstream: MStreamBuilder = MStream::builder(2, 1);
    let sketch: FeatureSketchBuilder = FeatureSketch::builder();

    assert_eq!(
        serde_json::to_value(
            forest
                .maybe_seed(None)
                .maybe_capacity(None)
                .build()
                .unwrap()
                .config()
        )
        .unwrap(),
        serde_json::to_value(RcfConfig::new(2)).unwrap(),
    );
    assert_eq!(
        serde_json::to_value(online.maybe_seed(None).build().unwrap().config()).unwrap(),
        serde_json::to_value(OnlineIForestConfig::new(2)).unwrap(),
    );
    assert_eq!(
        serde_json::to_value(mstream.maybe_seed(None).build().unwrap().config()).unwrap(),
        serde_json::to_value(MStreamConfig::new(2, 1)).unwrap(),
    );
    assert_eq!(
        serde_json::to_value(sketch.maybe_seed(None).build().unwrap().config()).unwrap(),
        serde_json::to_value(FeatureSketchConfig::new()).unwrap(),
    );
}

#[test]
fn public_config_builders_and_constant_defaults_are_available() {
    const CAPACITY: usize = RcfConfig::default_capacity();
    const WINDOW: usize = OnlineIForestConfig::default_window_size();
    const BUCKETS: usize = MStreamConfig::default_num_buckets();
    const HALF_LIFE: u64 = FeatureSketchConfig::default_decay_half_life();
    assert_eq!(
        (CAPACITY, WINDOW, BUCKETS, HALF_LIFE),
        (256, 2048, 1024, 2048)
    );

    let forest = RcfConfig::builder(2).capacity(64);
    let online = OnlineIForestConfig::builder(2).window_size(512);
    let mstream = MStreamConfig::builder(2, 1).num_buckets(256);
    let sketch = FeatureSketchConfig::builder().decay_half_life(16);
    assert_eq!(forest.build().capacity(), 64);
    assert_eq!(online.build().window_size(), 512);
    assert_eq!(mstream.build().num_buckets(), 256);
    assert_eq!(sketch.build().decay_half_life(), 16);
}

#[test]
#[allow(deprecated)]
fn deprecated_config_setters_still_overwrite_values() {
    assert_eq!(
        RcfConfig::new(2)
            .with_capacity(32)
            .with_capacity(64)
            .capacity(),
        RcfConfig::builder(2).capacity(64).build().capacity(),
    );
    assert_eq!(
        OnlineIForestConfig::new(2)
            .with_window_size(256)
            .with_window_size(512)
            .window_size(),
        OnlineIForestConfig::builder(2)
            .window_size(512)
            .build()
            .window_size(),
    );
    assert_eq!(
        MStreamConfig::new(2, 1)
            .with_num_buckets(128)
            .with_num_buckets(256)
            .num_buckets(),
        MStreamConfig::builder(2, 1)
            .num_buckets(256)
            .build()
            .num_buckets(),
    );
    assert_eq!(
        FeatureSketchConfig::new()
            .with_decay_half_life(8)
            .with_decay_half_life(16)
            .decay_half_life(),
        FeatureSketchConfig::builder()
            .decay_half_life(16)
            .build()
            .decay_half_life(),
    );
}

#[cfg(feature = "serde")]
#[test]
fn forest_builder_matches_seeded_config_construction() {
    let config = RcfConfig::builder(2)
        .shingle_size(3)
        .capacity(64)
        .num_trees(7)
        .time_decay(0.01)
        .output_after(5)
        .internal_shingling(false)
        .initial_accept_fraction(0.25)
        .build();
    let builder: ForestBuilder<_> = Forest::builder(2).seed(42);
    let builder = builder
        .shingle_size(3)
        .capacity(64)
        .num_trees(7)
        .time_decay(0.01)
        .output_after(5)
        .internal_shingling(false)
        .initial_accept_fraction(0.25);
    assert!(!format!("{builder:?}").is_empty());
    assert_eq!(
        serde_json::to_value(builder.clone().build().unwrap()).unwrap(),
        serde_json::to_value(Forest::from_config_seeded(&config, 42).unwrap()).unwrap(),
    );
}

#[cfg(feature = "serde")]
#[test]
fn onlineiforest_builder_matches_seeded_config_construction() {
    let config = OnlineIForestConfig::builder(2)
        .num_trees(7)
        .window_size(64)
        .max_leaf_samples(5)
        .build();
    let builder: OnlineIForestBuilder<_> = OnlineIForest::builder(2).maybe_seed(Some(42));
    let builder = builder.num_trees(7).window_size(64).max_leaf_samples(5);
    assert!(!format!("{builder:?}").is_empty());
    assert_eq!(
        serde_json::to_value(builder.clone().build().unwrap()).unwrap(),
        serde_json::to_value(OnlineIForest::from_config_seeded(&config, 42).unwrap()).unwrap(),
    );
}

#[cfg(feature = "serde")]
#[test]
fn mstream_builder_matches_seeded_config_construction() {
    let config = MStreamConfig::builder(2, 1)
        .num_rows(3)
        .num_buckets(32)
        .alpha(0.5)
        .build();
    let builder: MStreamBuilder<_> = MStream::builder(2, 1).maybe_seed(Some(42));
    let builder = builder.num_rows(3).num_buckets(32).alpha(0.5);
    assert!(!format!("{builder:?}").is_empty());
    assert_eq!(
        serde_json::to_value(builder.clone().build().unwrap()).unwrap(),
        serde_json::to_value(MStream::from_config_seeded(&config, 42).unwrap()).unwrap(),
    );
}

#[cfg(feature = "serde")]
#[test]
fn featuresketch_builder_matches_seeded_config_construction() {
    let config = FeatureSketchConfig::builder()
        .value_projection_dims(4)
        .presence_projection_dims(5)
        .chains_per_ensemble(2)
        .chain_depth(3)
        .sketch_rows(3)
        .sketch_buckets(32)
        .decay_half_life(16)
        .build();
    let builder: FeatureSketchBuilder<_> = FeatureSketch::builder().maybe_seed(Some(42));
    let builder = builder
        .value_projection_dims(4)
        .presence_projection_dims(5)
        .chains_per_ensemble(2)
        .chain_depth(3)
        .sketch_rows(3)
        .sketch_buckets(32)
        .decay_half_life(16);
    assert!(!format!("{builder:?}").is_empty());
    assert_eq!(
        serde_json::to_value(builder.clone().build().unwrap()).unwrap(),
        serde_json::to_value(FeatureSketch::from_config_seeded(&config, 42).unwrap()).unwrap(),
    );
}

#[rstest::rstest]
#[case::forest(Forest::builder(2).capacity(0).build().map(|_| ()))]
#[case::onlineiforest(OnlineIForest::builder(0).seed(42).build().map(|_| ()))]
#[case::mstream(MStream::builder(1, 0).alpha(f64::NAN).build().map(|_| ()))]
#[case::featuresketch(FeatureSketch::builder().chain_depth(0).seed(42).build().map(|_| ()))]
fn builders_reject_invalid_configuration(#[case] result: rcf3::Result<()>) {
    assert!(matches!(result, Err(RcfError::InvalidArgument(_))));
}
