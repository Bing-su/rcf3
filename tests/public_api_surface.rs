/// Smoke test for the crate-level public facade exports.
use rcf3::{
    Attribution, FeatureSketch, FeatureSketchBuilder, FeatureSketchConfig, Forest, ForestBuilder,
    NeighborResult, RcfConfig,
};

#[test]
fn top_level_facade_exports_expected_user_facing_types() {
    let config = RcfConfig::new(2)
        .with_shingle_size(3)
        .with_capacity(64)
        .with_num_trees(7)
        .with_time_decay(0.01)
        .with_output_after(5)
        .with_internal_shingling(false)
        .with_initial_accept_fraction(0.25);

    assert_eq!(config.input_dim(), 2);
    assert_eq!(config.shingle_size(), 3);
    assert_eq!(config.capacity(), 64);
    assert_eq!(config.num_trees(), 7);
    assert_eq!(config.time_decay(), 0.01);
    assert_eq!(config.output_after(), 5);
    assert!(!config.internal_shingling());
    assert_eq!(config.initial_accept_fraction(), 0.25);

    let builder: ForestBuilder = Forest::builder(2);
    let forest = builder.seed(7).build().unwrap();
    assert_eq!(forest.config().input_dim(), 2);

    let attr = Attribution {
        below: 1.25,
        above: 0.75,
    };
    assert_eq!(attr.total(), 2.0);

    let neighbor = NeighborResult {
        score: 0.5,
        point: vec![1.0, 2.0],
        distance: 3.0,
    };
    assert_eq!(neighbor.point, vec![1.0, 2.0]);

    let feature_config = FeatureSketchConfig::new()
        .with_value_projection_dims(4)
        .with_presence_projection_dims(4)
        .with_chains_per_ensemble(2)
        .with_chain_depth(2)
        .with_sketch_buckets(32);
    assert_eq!(feature_config.value_projection_dims(), 4);

    let builder: FeatureSketchBuilder = FeatureSketch::builder()
        .value_projection_dims(4)
        .presence_projection_dims(4)
        .chains_per_ensemble(2)
        .chain_depth(2)
        .sketch_buckets(32)
        .seed(7);
    let mut detector = builder.build().unwrap();
    detector.update([("feature", 1.0)]).unwrap();
    assert!(detector.score([("feature", 1.0)]).unwrap().is_finite());
}
