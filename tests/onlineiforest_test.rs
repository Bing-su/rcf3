use rcf3::OnlineIForest;

fn clustered_detector() -> OnlineIForest {
    let mut detector = OnlineIForest::builder(2)
        .num_trees(16)
        .window_size(64)
        .max_leaf_samples(4)
        .seed(42)
        .build()
        .unwrap();
    for idx in 0..96 {
        let dx = (idx % 8) as f32 * 0.02 - 0.07;
        let dy = (idx / 8 % 8) as f32 * 0.02 - 0.07;
        detector.update(&[dx, dy]).unwrap();
    }
    detector
}

#[test]
fn far_outlier_scores_above_cluster_point() {
    let detector = clustered_detector();
    let inlier = detector.score(&[0.0, 0.0]).unwrap();
    let outlier = detector.score(&[8.0, -8.0]).unwrap();
    assert!(outlier > inlier, "outlier={outlier} inlier={inlier}");
}

#[test]
fn sliding_window_adapts_to_new_region_after_drift() {
    let mut detector = OnlineIForest::builder(1)
        .num_trees(16)
        .window_size(32)
        .max_leaf_samples(4)
        .seed(7)
        .build()
        .unwrap();

    for idx in 0..32 {
        detector.update(&[(idx % 4) as f32 * 0.02]).unwrap();
    }
    let new_region_before = detector.score(&[10.02]).unwrap();

    for idx in 0..48 {
        detector.update(&[10.0 + (idx % 4) as f32 * 0.02]).unwrap();
    }

    let old_region_after = detector.score(&[0.02]).unwrap();
    let new_region_after = detector.score(&[10.02]).unwrap();
    assert!(
        new_region_after < new_region_before,
        "new region should become less anomalous after drift adaptation: before={new_region_before} after={new_region_after}"
    );
    assert!(
        old_region_after > new_region_after,
        "forgotten old region should become more anomalous than the current region: old={old_region_after} new={new_region_after}"
    );
}
