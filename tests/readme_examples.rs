// Test all code examples from README to ensure they compile and run correctly

#[cfg(test)]
mod readme_examples {
    use rcf3::Forest;

    #[test]
    fn test_creating_forest_basic() -> Result<(), Box<dyn std::error::Error>> {
        let forest = Forest::builder(2, 1) // 2D input, no shingling
            .num_trees(50)
            .capacity(256)
            .build()?;

        assert_eq!(forest.num_trees(), 50);
        Ok(())
    }

    #[test]
    fn test_creating_forest_with_time_series() -> Result<(), Box<dyn std::error::Error>> {
        let forest = Forest::builder(4, 8) // 4D input, window size 8
            .internal_shingling(true)
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
        let mut forest = Forest::builder(2, 1).capacity(256).num_trees(50).build()?;

        // Update the forest with a new observation
        let point = vec![1.5, 2.3];
        forest.update(&point)?;

        // Check if the forest has warmed up and get score
        if forest.is_ready() {
            let score = forest.score(&point)?;
            println!("Anomaly score: {}", score);
            assert!(score >= 0.0);
        }

        // Get the number of observations processed
        println!("Entries seen: {}", forest.entries_seen());

        Ok(())
    }

    #[test]
    fn test_scoring_methods() -> Result<(), Box<dyn std::error::Error>> {
        let mut forest = Forest::builder(3, 1).capacity(256).num_trees(50).build()?;

        // Feed some data to warm up the forest
        for _ in 0..100 {
            forest.update(&vec![1.5, 2.3, -0.5])?;
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
        let mut forest = Forest::builder(3, 1).capacity(256).num_trees(50).build()?;

        // Feed normal data first
        for _ in 0..100 {
            forest.update(&vec![1.0, 2.0, 3.0])?;
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
        let mut forest = Forest::builder(2, 1).capacity(256).num_trees(50).build()?;

        // Feed some data points to build up the forest
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
            forest.update(&point)?;
        }

        let query_point = vec![1.5, 2.3];
        let neighbors = forest.near_neighbors(&query_point, 3, 50)?;

        println!("Found {} neighbors:", neighbors.len());
        for neighbor in neighbors {
            println!(
                "Distance: {}, Score: {}, Point: {:?}",
                neighbor.distance, neighbor.score, neighbor.point
            );
        }

        Ok(())
    }

    #[test]
    fn test_missing_value_imputation() -> Result<(), Box<dyn std::error::Error>> {
        let mut forest = Forest::builder(3, 1).capacity(256).num_trees(50).build()?;

        // Feed some complete data to train
        for i in 0..100 {
            forest.update(&vec![1.0 + (i as f32) * 0.01, 2.0, 3.0])?;
        }

        // Test imputation with a missing value at index 1
        let point = vec![1.5, f32::NAN, 3.0];
        let missing = vec![1];
        let imputed = forest.impute(&point, &missing, 1.0)?;

        println!("Imputed value at index 1: {}", imputed[1]);
        assert!(!imputed[1].is_nan());
        assert!(imputed.len() == 3);

        Ok(())
    }

    #[test]
    #[cfg(feature = "serde")]
    fn test_serialization() -> Result<(), Box<dyn std::error::Error>> {
        let mut forest = Forest::builder(2, 1).capacity(256).num_trees(50).build()?;

        // Feed some data
        for _ in 0..50 {
            forest.update(&vec![1.5, 2.3])?;
        }

        // Save to string
        let json_str = forest.to_json()?;
        assert!(!json_str.is_empty());
        println!("JSON length: {}", json_str.len());

        // Load from string
        let loaded = Forest::from_json(&json_str)?;
        assert_eq!(loaded.num_trees(), forest.num_trees());

        Ok(())
    }

    #[test]
    fn test_anomaly_detection_example() -> Result<(), Box<dyn std::error::Error>> {
        let mut forest = Forest::builder(3, 1).capacity(256).num_trees(50).build()?;

        // Warm up the forest with many normal data points
        for i in 0..200 {
            let val = (i as f32) * 0.01;
            forest.update(&vec![1.0 + val, 2.0 + val, 3.0 + val])?;
        }

        let data = vec![
            vec![1.0, 2.0, 3.0],
            vec![1.1, 2.1, 3.1],
            vec![1.2, 2.2, 3.2],
            vec![100.0, 200.0, 300.0], // Extreme anomaly
            vec![1.3, 2.3, 3.3],
        ];

        let mut anomaly_count = 0;
        for point in data {
            // Online inference order: score first, then update.
            if forest.is_ready() {
                let score = forest.score(&point)?;
                let attribution = forest.attribution(&point)?;

                println!("Point: {:?}, Score: {}", point, score);

                // Lower threshold since we're detecting a very extreme anomaly
                if score > 0.1 {
                    println!("Anomaly detected: score={}", score);
                    for (i, attr) in attribution.iter().enumerate() {
                        println!("  Dimension {}: {:.2}", i, attr.above);
                    }
                    anomaly_count += 1;
                }
            }

            forest.update(&point)?;
        }

        println!("Total anomalies detected: {}", anomaly_count);
        assert!(anomaly_count > 0); // We expect to detect the anomaly

        Ok(())
    }

    #[test]
    fn test_time_series_forecasting() -> Result<(), Box<dyn std::error::Error>> {
        let mut forest = Forest::builder(4, 8)
            .internal_shingling(true)
            .capacity(512)
            .num_trees(50)
            .build()?;

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

        // Predict the next 5 observations
        if forest.is_ready() {
            let predictions = forest.extrapolate(5)?;
            println!("Predictions (flat list): {:?}", predictions);
            // Returns a flat list of length 5 * input_dim = 5 * 4 = 20
            assert_eq!(predictions.len(), 20);
        }

        Ok(())
    }
}
