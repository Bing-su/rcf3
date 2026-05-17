// Test the executable documentation examples to keep the public guides honest.

#[cfg(test)]
mod docs_examples {
    use rcf3::{Forest, MStream};

    #[test]
    fn test_creating_forest_basic() -> Result<(), Box<dyn std::error::Error>> {
        let forest = Forest::builder(2) // 2D input, no shingling
            .shingle_size(1)
            .num_trees(50)
            .capacity(256)
            .build()?;

        assert_eq!(forest.num_trees(), 50);
        Ok(())
    }

    #[test]
    fn test_creating_forest_with_time_series() -> Result<(), Box<dyn std::error::Error>> {
        let forest = Forest::builder(4)
            .shingle_size(8)
            .num_trees(100)
            .capacity(512)
            .time_decay(0.01)
            .build()?;

        assert_eq!(forest.num_trees(), 100);
        Ok(())
    }

    #[test]
    fn test_creating_forest_from_config() -> Result<(), Box<dyn std::error::Error>> {
        use rcf3::{Forest, RcfConfig};

        let config = RcfConfig::new(3)
            .with_num_trees(75)
            .with_capacity(512)
            .with_shingle_size(4);

        let forest = Forest::from_config(&config)?;
        assert_eq!(forest.num_trees(), 75);
        Ok(())
    }

    #[test]
    fn test_basic_operations() -> Result<(), Box<dyn std::error::Error>> {
        let mut forest = Forest::builder(2)
            .shingle_size(1)
            .capacity(256)
            .num_trees(50)
            .build()?;

        let point = vec![1.5, 2.3];

        if forest.is_ready() {
            let score = forest.score(&point)?;
            println!("Anomaly score: {score}");
            assert!(score >= 0.0);
        }

        forest.update(&point)?;
        println!("Entries seen: {}", forest.entries_seen());

        Ok(())
    }

    #[test]
    fn test_scoring_methods() -> Result<(), Box<dyn std::error::Error>> {
        let mut forest = Forest::builder(3)
            .shingle_size(1)
            .capacity(256)
            .num_trees(50)
            .build()?;

        // Feed some data to warm up the forest
        for _ in 0..100 {
            forest.update(&[1.5, 2.3, -0.5])?;
        }

        let point = vec![1.5, 2.3, -0.5];

        // Anomaly Score (RCF Score)
        let score = forest.score(&point)?;
        assert!(score >= 0.0);
        println!("RCF Score: {}", score);

        // Displacement Score
        let displacement = forest.displacement_score(&point)?;
        assert!(displacement >= 0.0);
        println!("Displacement Score: {}", displacement);

        // Density Estimate
        let density = forest.density(&point)?;
        assert!(density >= 0.0);
        println!("Density: {}", density);

        Ok(())
    }

    #[test]
    fn test_feature_attribution() -> Result<(), Box<dyn std::error::Error>> {
        let mut forest = Forest::builder(3)
            .shingle_size(1)
            .capacity(256)
            .num_trees(50)
            .build()?;

        // Feed normal data first
        for _ in 0..100 {
            forest.update(&[1.0, 2.0, 3.0])?;
        }

        // Test with anomalous point
        let point = vec![1.5, 2.3, 100.0]; // Third dimension is anomalous
        let attribution = forest.attribution(&point)?;

        for (i, attr) in attribution.iter().enumerate() {
            println!(
                "Dimension {}: below={}, above={}",
                i, attr.below, attr.above
            );
            assert!(attr.below >= 0.0);
            assert!(attr.above >= 0.0);
        }

        Ok(())
    }

    #[test]
    fn test_neighborhood_search() -> Result<(), Box<dyn std::error::Error>> {
        let mut forest = Forest::builder(2)
            .shingle_size(1)
            .capacity(256)
            .num_trees(50)
            .build()?;

        let data = vec![
            vec![1.0, 2.0],
            vec![1.1, 2.1],
            vec![1.2, 2.2],
            vec![1.3, 2.3],
            vec![1.4, 2.4],
            vec![5.0, 6.0],
            vec![5.1, 6.1],
            vec![5.2, 6.2],
        ];

        for point in &data {
            forest.update(point)?;
        }

        let neighbors = forest.near_neighbors(&[1.5, 2.3], 10, 50)?;
        for neighbor in neighbors {
            println!("distance={}, score={}", neighbor.distance, neighbor.score);
        }

        Ok(())
    }

    #[test]
    fn test_missing_value_imputation() -> Result<(), Box<dyn std::error::Error>> {
        let mut forest = Forest::builder(3).build()?;

        // Feed some complete data to train
        for i in 0..100 {
            forest.update(&[1.0 + (i as f32) * 0.01, 2.0, 3.0])?;
        }

        let point = vec![1.5, f32::NAN, 3.0];
        let missing = vec![1];
        let imputed = forest.impute(&point, &missing, 1.0)?;

        assert!(!imputed[1].is_nan());
        assert!(imputed.len() == 3);

        Ok(())
    }

    #[test]
    #[cfg(feature = "serde")]
    fn test_serialization() -> Result<(), Box<dyn std::error::Error>> {
        let mut forest = Forest::builder(2).build()?;

        for _ in 0..50 {
            forest.update(&[1.5, 2.3])?;
        }

        let json_str = forest.to_json()?;
        assert!(!json_str.is_empty());
        let tmpdir = tempfile::tempdir()?;
        let path = tmpdir.path().join("forest.json");
        forest.save_json(&path)?;

        let loaded = Forest::from_json(&json_str)?;
        let loaded_from_file = Forest::load_json(&path)?;
        assert_eq!(loaded.num_trees(), forest.num_trees());
        assert_eq!(loaded_from_file.num_trees(), forest.num_trees());

        Ok(())
    }

    #[test]
    fn test_anomaly_detection_example() -> Result<(), Box<dyn std::error::Error>> {
        let mut forest = Forest::builder(3).capacity(256).num_trees(50).build()?;

        // Warm up the forest with many normal data points
        for i in 0..200 {
            let val = (i as f32) * 0.01;
            forest.update(&[1.0 + val, 2.0 + val, 3.0 + val])?;
        }

        let data = vec![
            vec![1.0, 2.0, 3.0],
            vec![1.1, 2.1, 3.1],
            vec![100.0, 200.0, 300.0],
        ];

        for point in data {
            if forest.is_ready() {
                let score = forest.score(&point)?;
                println!("Point: {point:?}, score={score}");
            }

            forest.update(&point)?;
        }

        Ok(())
    }

    #[test]
    fn test_time_series_forecasting() -> Result<(), Box<dyn std::error::Error>> {
        let mut forest = Forest::builder(4).shingle_size(8).build()?;

        // Feed observations one at a time (time series)
        let stream = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![1.1, 2.1, 3.1, 4.1],
            vec![1.2, 2.2, 3.2, 4.2],
            vec![1.3, 2.3, 3.3, 4.3],
            vec![1.4, 2.4, 3.4, 4.4],
            vec![1.5, 2.5, 3.5, 4.5],
            vec![1.6, 2.6, 3.6, 4.6],
            vec![1.7, 2.7, 3.7, 4.7],
            vec![1.8, 2.8, 3.8, 4.8],
            vec![1.9, 2.9, 3.9, 4.9],
        ];

        for point in stream {
            forest.update(&point)?;
        }

        let predictions = forest.extrapolate(5)?;
        assert_eq!(predictions.len(), 20);

        Ok(())
    }

    #[test]
    fn test_mstream_basic_usage() -> Result<(), Box<dyn std::error::Error>> {
        let mut detector = MStream::builder(2, 1)
            .alpha(0.8)
            .num_rows(2)
            .num_buckets(1024)
            .seed(7)
            .build()?;

        let score = detector.update_and_score(&[1.5, 2.0], &[7], 1)?;
        assert!(score >= 0.0);
        Ok(())
    }

    #[test]
    fn test_mstream_preview_and_detailed_scores() -> Result<(), Box<dyn std::error::Error>> {
        let mut detector = MStream::builder(2, 1).seed(7).build()?;
        detector.update(&[1.5, 2.0], &[7], 1)?;

        let preview = detector.score(&[1.5, 2.0], &[7], 2)?;
        let committed = detector.update_and_score(&[1.5, 2.0], &[7], 2)?;
        assert_eq!(preview, committed);

        let detailed = detector.score_detailed(&[1.5, 2.0], &[7], 3)?;
        assert_eq!(detailed.numeric_features.len(), 2);
        assert_eq!(detailed.categorical_features.len(), 1);
        assert!(detector.is_ready());
        assert_eq!(detector.entries_seen(), 2);
        assert_eq!(detector.current_time(), Some(2));
        Ok(())
    }

    #[test]
    #[cfg(feature = "serde")]
    fn test_mstream_serialization() -> Result<(), Box<dyn std::error::Error>> {
        let mut detector = MStream::builder(2, 1).seed(7).build()?;
        detector.update(&[1.5, 2.0], &[7], 1)?;

        let json = detector.to_json()?;
        let restored = MStream::from_json(json)?;
        let tmpdir = tempfile::tempdir()?;
        let path = tmpdir.path().join("mstream.json");
        detector.save_json(&path)?;
        let restored_from_file = MStream::load_json(&path)?;
        assert_eq!(restored.entries_seen(), detector.entries_seen());
        assert_eq!(restored.current_time(), detector.current_time());
        assert_eq!(restored_from_file.entries_seen(), detector.entries_seen());
        assert_eq!(restored_from_file.current_time(), detector.current_time());
        Ok(())
    }

    #[test]
    fn test_mstream_practical_example() -> Result<(), Box<dyn std::error::Error>> {
        let mut detector = MStream::builder(2, 2).seed(2026).num_buckets(512).build()?;

        let normal = detector.update_and_score(&[0.0, 3.2], &[1, 10], 1)?;
        let suspicious = detector.score_detailed(&[12.0, 0.3], &[99, 10], 2)?;

        println!("normal={normal}, suspicious={}", suspicious.total);
        println!(
            "failed-attempt contribution={}",
            suspicious.numeric_features[0]
        );
        println!(
            "country contribution={}",
            suspicious.categorical_features[0]
        );
        Ok(())
    }
}
