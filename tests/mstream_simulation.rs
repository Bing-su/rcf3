//! End-to-end mStream simulations through the public API.

use approx::abs_diff_eq;
use fake::rand::prelude::*;
use fake::{Dummy, Fake, Faker};
use proptest::prelude::*;
use rcf3::MStream;

mod fixtures {
    use super::*;

    #[derive(Clone, Copy)]
    pub(super) struct LoginEvent {
        failed_attempts: f64,
        bytes_sent_kib: f64,
        country_id: i64,
        endpoint_id: i64,
        tick: u64,
    }

    impl LoginEvent {
        pub(super) fn numeric(self) -> [f64; 2] {
            [self.failed_attempts, self.bytes_sent_kib]
        }

        pub(super) fn categorical(self) -> [i64; 2] {
            [self.country_id, self.endpoint_id]
        }

        pub(super) fn tick(self) -> u64 {
            self.tick
        }
    }

    #[derive(Debug, Dummy)]
    struct NormalLoginSample {
        #[dummy(faker = "0..2")]
        failed_attempts: u8,
        #[dummy(faker = "30..36")]
        bytes_sent_tenths_kib: u8,
        #[dummy(faker = "1..3")]
        country_id: u8,
    }

    pub(super) struct LoginEventFactory {
        rng: StdRng,
    }

    impl LoginEventFactory {
        pub(super) fn seeded(seed: u64) -> Self {
            Self {
                rng: StdRng::seed_from_u64(seed),
            }
        }

        pub(super) fn normal_login(&mut self, tick: u64) -> LoginEvent {
            let sample: NormalLoginSample = Faker.fake_with_rng(&mut self.rng);

            LoginEvent {
                failed_attempts: sample.failed_attempts as f64,
                bytes_sent_kib: sample.bytes_sent_tenths_kib as f64 / 10.0,
                country_id: sample.country_id as i64,
                endpoint_id: 10,
                tick,
            }
        }

        pub(super) fn credential_stuffing_attempt(&self, tick: u64) -> LoginEvent {
            LoginEvent {
                failed_attempts: 12.0,
                bytes_sent_kib: 0.3,
                country_id: 99,
                endpoint_id: 10,
                tick,
            }
        }

        pub(super) fn familiar_login(&self, tick: u64) -> LoginEvent {
            LoginEvent {
                failed_attempts: 0.0,
                bytes_sent_kib: 3.2,
                country_id: 1,
                endpoint_id: 10,
                tick,
            }
        }

        pub(super) fn exfiltration_attempt(&self, tick: u64) -> LoginEvent {
            LoginEvent {
                failed_attempts: 0.0,
                bytes_sent_kib: 512.0,
                country_id: 2,
                endpoint_id: 42,
                tick,
            }
        }

        pub(super) fn unfamiliar_route_login(&self, tick: u64) -> LoginEvent {
            LoginEvent {
                failed_attempts: 0.0,
                bytes_sent_kib: 3.2,
                country_id: 3,
                endpoint_id: 11,
                tick,
            }
        }
    }
}

use fixtures::{LoginEvent, LoginEventFactory};

fn detector(seed: u64) -> MStream {
    MStream::builder(2, 2)
        .seed(seed)
        .alpha(0.8)
        .num_rows(2)
        .num_buckets(512)
        .build()
        .unwrap()
}

fn train_on_normal_traffic(detector: &mut MStream, factory: &mut LoginEventFactory) -> Vec<f64> {
    let mut scores = Vec::new();
    for tick in 1..=12 {
        for _ in 0..8 {
            let event = factory.normal_login(tick);
            scores.push(commit(detector, event));
        }
    }
    scores
}

fn commit(detector: &mut MStream, event: LoginEvent) -> f64 {
    detector
        .update_and_score(&event.numeric(), &event.categorical(), event.tick())
        .unwrap()
}

fn peak(scores: impl IntoIterator<Item = f64>) -> f64 {
    scores.into_iter().fold(0.0_f64, f64::max)
}

fn assert_credential_stuffing_burst_scores_above_normal_traffic(
    traffic_seed: u64,
    detector_seed: u64,
) -> Result<(), TestCaseError> {
    let mut factory = LoginEventFactory::seeded(traffic_seed);
    let mut detector = detector(detector_seed);

    let normal_peak = peak(train_on_normal_traffic(&mut detector, &mut factory));
    let attack_peak =
        peak((0..8).map(|_| commit(&mut detector, factory.credential_stuffing_attempt(13))));

    prop_assert!(
        attack_peak > normal_peak,
        "attack peak should exceed normal peak: traffic_seed={traffic_seed}, detector_seed={detector_seed}, attack={attack_peak}, normal={normal_peak}"
    );

    Ok(())
}

fn assert_attack_preview_highlights_the_shift_before_committing_it(
    traffic_seed: u64,
    detector_seed: u64,
) -> Result<(), TestCaseError> {
    let mut factory = LoginEventFactory::seeded(traffic_seed);
    let mut detector = detector(detector_seed);
    train_on_normal_traffic(&mut detector, &mut factory);

    let attack = factory.credential_stuffing_attempt(13);
    let preview = detector
        .score_detailed(&attack.numeric(), &attack.categorical(), attack.tick())
        .unwrap();
    let before_commit_entries = detector.entries_seen();
    let committed = detector
        .update_and_score_detailed(&attack.numeric(), &attack.categorical(), attack.tick())
        .unwrap();

    prop_assert_eq!(detector.entries_seen(), before_commit_entries + 1);
    prop_assert!(
        abs_diff_eq!(preview.total, committed.total, epsilon = 1e-12),
        "preview should match committed score: traffic_seed={traffic_seed}, detector_seed={detector_seed}, preview={}, committed={}",
        preview.total,
        committed.total
    );
    prop_assert!(
        preview.numeric_features[0] > 0.0,
        "failed-attempt spike should contribute to the anomaly score: traffic_seed={traffic_seed}, detector_seed={detector_seed}, contribution={}",
        preview.numeric_features[0]
    );
    prop_assert!(
        preview.categorical_features[0] > 0.0,
        "unseen country should contribute to the anomaly score: traffic_seed={traffic_seed}, detector_seed={detector_seed}, contribution={}",
        preview.categorical_features[0]
    );

    Ok(())
}

fn assert_large_data_exfiltration_highlights_bytes_sent(
    traffic_seed: u64,
    detector_seed: u64,
) -> Result<(), TestCaseError> {
    let mut factory = LoginEventFactory::seeded(traffic_seed);
    let mut detector = detector(detector_seed);

    train_on_normal_traffic(&mut detector, &mut factory);
    let exfiltration = factory.exfiltration_attempt(13);
    let exfiltration_preview = detector
        .score_detailed(
            &exfiltration.numeric(),
            &exfiltration.categorical(),
            exfiltration.tick(),
        )
        .unwrap();

    prop_assert!(exfiltration_preview.total.is_finite());
    prop_assert!(exfiltration_preview.total >= 0.0);
    prop_assert!(
        exfiltration_preview.numeric_features[1] > 0.0,
        "large transfer should contribute to the bytes-sent feature score: traffic_seed={traffic_seed}, detector_seed={detector_seed}, contribution={}",
        exfiltration_preview.numeric_features[1]
    );
    prop_assert!(
        exfiltration_preview
            .numeric_features
            .iter()
            .chain(&exfiltration_preview.categorical_features)
            .all(|score| score.is_finite() && *score >= 0.0),
        "detailed exfiltration score should contain only finite non-negative contributions: traffic_seed={traffic_seed}, detector_seed={detector_seed}, score={exfiltration_preview:?}"
    );

    Ok(())
}

fn assert_unfamiliar_route_burst_highlights_route_features(
    traffic_seed: u64,
    detector_seed: u64,
) -> Result<(), TestCaseError> {
    let mut factory = LoginEventFactory::seeded(traffic_seed);
    let mut detector = detector(detector_seed);
    train_on_normal_traffic(&mut detector, &mut factory);

    let mut familiar_detector = detector.clone();
    let unfamiliar_event = factory.unfamiliar_route_login(13);
    let unfamiliar_preview = detector
        .score_detailed(
            &unfamiliar_event.numeric(),
            &unfamiliar_event.categorical(),
            unfamiliar_event.tick(),
        )
        .unwrap();
    let mut unfamiliar_detector = detector;
    let familiar_peak =
        peak((0..8).map(|_| commit(&mut familiar_detector, factory.familiar_login(13))));
    let unfamiliar_peak =
        peak((0..8).map(|_| commit(&mut unfamiliar_detector, factory.unfamiliar_route_login(13))));

    prop_assert!(familiar_peak.is_finite());
    prop_assert!(unfamiliar_peak.is_finite());
    prop_assert!(familiar_peak >= 0.0);
    prop_assert!(
        unfamiliar_peak >= 0.0,
        "burst peaks should be finite and non-negative: traffic_seed={traffic_seed}, detector_seed={detector_seed}, unfamiliar={unfamiliar_peak}, familiar={familiar_peak}"
    );
    prop_assert!(
        unfamiliar_preview.categorical_features[0] > 0.0,
        "unseen country should contribute to unfamiliar-route score: traffic_seed={traffic_seed}, detector_seed={detector_seed}, contribution={}",
        unfamiliar_preview.categorical_features[0]
    );
    prop_assert!(
        unfamiliar_preview.categorical_features[1] > 0.0,
        "unseen endpoint should contribute to unfamiliar-route score: traffic_seed={traffic_seed}, detector_seed={detector_seed}, contribution={}",
        unfamiliar_preview.categorical_features[1]
    );

    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    #[test]
    fn credential_stuffing_burst_scores_above_normal_traffic(
        traffic_seed in any::<u64>(),
        detector_seed in any::<u64>(),
    ) {
        assert_credential_stuffing_burst_scores_above_normal_traffic(traffic_seed, detector_seed)?;
    }

    #[test]
    fn attack_preview_highlights_the_shift_before_committing_it(
        traffic_seed in any::<u64>(),
        detector_seed in any::<u64>(),
    ) {
        assert_attack_preview_highlights_the_shift_before_committing_it(traffic_seed, detector_seed)?;
    }

    #[test]
    fn large_data_exfiltration_highlights_bytes_sent(
        traffic_seed in any::<u64>(),
        detector_seed in any::<u64>(),
    ) {
        assert_large_data_exfiltration_highlights_bytes_sent(traffic_seed, detector_seed)?;
    }

    #[test]
    fn unfamiliar_route_burst_highlights_route_features(
        traffic_seed in any::<u64>(),
        detector_seed in any::<u64>(),
    ) {
        assert_unfamiliar_route_burst_highlights_route_features(traffic_seed, detector_seed)?;
    }
}
