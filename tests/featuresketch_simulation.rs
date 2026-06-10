//! End-to-end FeatureSketch simulations through the public API.

use fake::rand::prelude::*;
use fake::{Dummy, Fake, Faker};
use proptest::prelude::*;
use rcf3::FeatureSketch;

mod fixtures {
    use super::*;

    #[derive(Clone)]
    pub(super) struct ApiEvent {
        endpoint: &'static str,
        method: &'static str,
        status: u16,
        authenticated: bool,
        known_device: bool,
        mfa_present: bool,
        latency_ms: f64,
        bytes_out_kib: f64,
        failed_auths: f64,
        extra_features: Vec<(&'static str, f64)>,
    }

    impl ApiEvent {
        pub(super) fn features(&self) -> Vec<(String, f64)> {
            let mut features = vec![
                (format!("endpoint:{}", self.endpoint), 1.0),
                (format!("method:{}", self.method), 1.0),
                (format!("status:{}", self.status), 1.0),
                ("service:payments-api".to_string(), 1.0),
                ("latency_ms".to_string(), self.latency_ms),
                ("bytes_out_kib".to_string(), self.bytes_out_kib),
                ("failed_auths".to_string(), self.failed_auths),
            ];
            if self.authenticated {
                features.push(("auth:authenticated".to_string(), 1.0));
            }
            if self.known_device {
                features.push(("device:known".to_string(), 1.0));
            }
            if self.mfa_present {
                features.push(("mfa:present".to_string(), 1.0));
            }
            for (name, value) in &self.extra_features {
                features.push(((*name).to_string(), *value));
            }
            features
        }
    }

    #[derive(Debug, Dummy)]
    struct NormalApiSample {
        #[dummy(faker = "0..3")]
        endpoint_id: u8,
        #[dummy(faker = "0..2")]
        method_id: u8,
        #[dummy(faker = "38..62")]
        latency_ms: u16,
        #[dummy(faker = "18..42")]
        bytes_out_tenths_kib: u16,
    }

    pub(super) struct ApiEventFactory {
        rng: StdRng,
    }

    impl ApiEventFactory {
        pub(super) fn seeded(seed: u64) -> Self {
            Self {
                rng: StdRng::seed_from_u64(seed),
            }
        }

        pub(super) fn normal_request(&mut self) -> ApiEvent {
            let sample: NormalApiSample = Faker.fake_with_rng(&mut self.rng);
            let endpoint = match sample.endpoint_id {
                0 => "/login",
                1 => "/checkout",
                _ => "/account",
            };
            let method = if sample.method_id == 0 { "GET" } else { "POST" };

            ApiEvent {
                endpoint,
                method,
                status: 200,
                authenticated: true,
                known_device: true,
                mfa_present: true,
                latency_ms: f64::from(sample.latency_ms),
                bytes_out_kib: f64::from(sample.bytes_out_tenths_kib) / 10.0,
                failed_auths: 0.0,
                extra_features: Vec::new(),
            }
        }

        pub(super) fn familiar_request(&self) -> ApiEvent {
            ApiEvent {
                endpoint: "/checkout",
                method: "POST",
                status: 200,
                authenticated: true,
                known_device: true,
                mfa_present: true,
                latency_ms: 48.0,
                bytes_out_kib: 3.0,
                failed_auths: 0.0,
                extra_features: Vec::new(),
            }
        }

        pub(super) fn admin_probe(&self) -> ApiEvent {
            ApiEvent {
                endpoint: "/admin/export",
                method: "POST",
                status: 403,
                authenticated: true,
                known_device: false,
                mfa_present: false,
                latency_ms: 180.0,
                bytes_out_kib: 96.0,
                failed_auths: 6.0,
                extra_features: vec![
                    ("header:x-forwarded-admin", 1.0),
                    ("role:attempted-root", 1.0),
                    ("scope:bulk-export", 1.0),
                ],
            }
        }

        pub(super) fn missing_security_context(&self) -> ApiEvent {
            ApiEvent {
                endpoint: "/checkout",
                method: "POST",
                status: 200,
                authenticated: true,
                known_device: false,
                mfa_present: false,
                latency_ms: 50.0,
                bytes_out_kib: 3.1,
                failed_auths: 0.0,
                extra_features: Vec::new(),
            }
        }
    }
}

use fixtures::{ApiEvent, ApiEventFactory};

fn detector(seed: u64) -> FeatureSketch {
    FeatureSketch::builder()
        .value_projection_dims(16)
        .presence_projection_dims(16)
        .chains_per_ensemble(12)
        .chain_depth(6)
        .sketch_rows(2)
        .sketch_buckets(512)
        .decay_half_life(512)
        .seed(seed)
        .build()
        .unwrap()
}

fn train_on_normal_traffic(
    detector: &mut FeatureSketch,
    factory: &mut ApiEventFactory,
) -> Vec<f64> {
    let mut scores = Vec::new();
    for _ in 0..512 {
        scores.push(score_then_update(detector, factory.normal_request()));
    }
    scores
}

fn score_then_update(detector: &mut FeatureSketch, event: ApiEvent) -> f64 {
    let features = event.features();
    let score = detector
        .score(features.iter().map(|(name, value)| (name.as_str(), *value)))
        .unwrap();
    detector
        .update(features.iter().map(|(name, value)| (name.as_str(), *value)))
        .unwrap();
    score
}

fn p95(mut scores: Vec<f64>) -> f64 {
    scores.sort_by(|left, right| left.partial_cmp(right).unwrap());
    scores[((scores.len() * 95) / 100).min(scores.len() - 1)]
}

fn peak(scores: impl IntoIterator<Item = f64>) -> f64 {
    scores.into_iter().fold(0.0_f64, f64::max)
}

fn assert_admin_probe_scores_above_normal_api_traffic(
    traffic_seed: u64,
    detector_seed: u64,
) -> Result<(), TestCaseError> {
    let mut factory = ApiEventFactory::seeded(traffic_seed);
    let mut detector = detector(detector_seed);

    train_on_normal_traffic(&mut detector, &mut factory);
    let normal_scores: Vec<_> = (0..64)
        .map(|_| score_then_update(&mut detector, factory.normal_request()))
        .collect();
    let normal_p95 = p95(normal_scores);
    let probe_score = detector.score(factory.admin_probe().features()).unwrap();

    prop_assert!(
        probe_score > normal_p95,
        "admin probe should exceed normal p95: traffic_seed={traffic_seed}, detector_seed={detector_seed}, probe={probe_score}, normal_p95={normal_p95}"
    );

    Ok(())
}

fn assert_missing_security_context_scores_above_familiar_request(
    traffic_seed: u64,
    detector_seed: u64,
) -> Result<(), TestCaseError> {
    let mut factory = ApiEventFactory::seeded(traffic_seed);
    let mut detector = detector(detector_seed);
    train_on_normal_traffic(&mut detector, &mut factory);

    let familiar = detector
        .score(factory.familiar_request().features())
        .unwrap();
    let missing_context = detector
        .score(factory.missing_security_context().features())
        .unwrap();

    prop_assert!(
        missing_context > familiar,
        "missing security context should score above a familiar request: traffic_seed={traffic_seed}, detector_seed={detector_seed}, missing={missing_context}, familiar={familiar}"
    );

    Ok(())
}

fn assert_repeated_admin_probe_adapts_after_training(
    traffic_seed: u64,
    detector_seed: u64,
) -> Result<(), TestCaseError> {
    let mut factory = ApiEventFactory::seeded(traffic_seed);
    let mut detector = detector(detector_seed);
    train_on_normal_traffic(&mut detector, &mut factory);

    let first_score = detector.score(factory.admin_probe().features()).unwrap();
    for _ in 0..512 {
        detector.update(factory.admin_probe().features()).unwrap();
    }
    let adapted_score = detector.score(factory.admin_probe().features()).unwrap();

    prop_assert!(
        adapted_score < first_score,
        "repeated probe pattern should adapt downward: traffic_seed={traffic_seed}, detector_seed={detector_seed}, first={first_score}, adapted={adapted_score}"
    );

    Ok(())
}

fn assert_admin_probe_burst_scores_above_familiar_burst(
    traffic_seed: u64,
    detector_seed: u64,
) -> Result<(), TestCaseError> {
    let mut factory = ApiEventFactory::seeded(traffic_seed);
    let mut detector = detector(detector_seed);
    train_on_normal_traffic(&mut detector, &mut factory);

    let mut familiar_detector = detector.clone();
    let mut probe_detector = detector;
    let familiar_peak =
        peak((0..8).map(|_| score_then_update(&mut familiar_detector, factory.familiar_request())));
    let probe_peak =
        peak((0..8).map(|_| score_then_update(&mut probe_detector, factory.admin_probe())));

    prop_assert!(
        probe_peak > familiar_peak,
        "probe burst should exceed familiar burst: traffic_seed={traffic_seed}, detector_seed={detector_seed}, probe={probe_peak}, familiar={familiar_peak}"
    );

    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    #[test]
    fn admin_probe_scores_above_normal_api_traffic(
        traffic_seed in any::<u64>(),
        detector_seed in any::<u64>(),
    ) {
        assert_admin_probe_scores_above_normal_api_traffic(traffic_seed, detector_seed)?;
    }

    #[test]
    fn missing_security_context_scores_above_familiar_request(
        traffic_seed in any::<u64>(),
        detector_seed in any::<u64>(),
    ) {
        assert_missing_security_context_scores_above_familiar_request(traffic_seed, detector_seed)?;
    }

    #[test]
    fn repeated_admin_probe_adapts_after_training(
        traffic_seed in any::<u64>(),
        detector_seed in any::<u64>(),
    ) {
        assert_repeated_admin_probe_adapts_after_training(traffic_seed, detector_seed)?;
    }

    #[test]
    fn admin_probe_burst_scores_above_familiar_burst(
        traffic_seed in any::<u64>(),
        detector_seed in any::<u64>(),
    ) {
        assert_admin_probe_burst_scores_above_familiar_burst(traffic_seed, detector_seed)?;
    }
}
