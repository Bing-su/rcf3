//! End-to-end OnlineIForest simulations through the public API.

use fake::rand::prelude::*;
use fake::{Dummy, Fake, Faker};
use rcf3::OnlineIForest;

mod fixtures {
    use super::*;

    #[derive(Clone, Copy)]
    pub(super) struct CheckoutEvent {
        amount_usd: f32,
        session_seconds: f32,
        failed_payment_ratio: f32,
    }

    impl CheckoutEvent {
        pub(super) fn features(self) -> [f32; 3] {
            [
                self.amount_usd,
                self.session_seconds,
                self.failed_payment_ratio,
            ]
        }
    }

    #[derive(Debug, Dummy)]
    struct NormalCheckoutSample {
        #[dummy(faker = "90..116")]
        amount_dollars: u16,
        #[dummy(faker = "40..58")]
        session_seconds: u16,
        #[dummy(faker = "0..5")]
        failed_payment_basis_points: u16,
    }

    pub(super) struct CheckoutEventFactory {
        rng: StdRng,
    }

    impl CheckoutEventFactory {
        pub(super) fn seeded(seed: u64) -> Self {
            Self {
                rng: StdRng::seed_from_u64(seed),
            }
        }

        pub(super) fn normal_checkout(&mut self) -> CheckoutEvent {
            let sample: NormalCheckoutSample = Faker.fake_with_rng(&mut self.rng);

            CheckoutEvent {
                amount_usd: sample.amount_dollars as f32,
                session_seconds: sample.session_seconds as f32,
                failed_payment_ratio: sample.failed_payment_basis_points as f32 / 10_000.0,
            }
        }

        pub(super) fn account_takeover_attempt(&self, idx: usize) -> CheckoutEvent {
            CheckoutEvent {
                amount_usd: 2_000.0 + (idx % 3) as f32 * 50.0,
                session_seconds: 1_000.0 + (idx % 4) as f32 * 25.0,
                failed_payment_ratio: 0.95 + (idx % 2) as f32 * 0.02,
            }
        }
    }
}

use fixtures::{CheckoutEvent, CheckoutEventFactory};

fn detector() -> OnlineIForest {
    OnlineIForest::builder(3)
        .num_trees(64)
        .window_size(256)
        .max_leaf_samples(8)
        .seed(2026)
        .build()
        .unwrap()
}

fn train_on_normal_checkouts(detector: &mut OnlineIForest, factory: &mut CheckoutEventFactory) {
    for _ in 0..512 {
        score_then_update(detector, factory.normal_checkout());
    }
}

fn score_then_update(detector: &mut OnlineIForest, event: CheckoutEvent) -> f64 {
    let features = event.features();
    let score = detector.score(&features).unwrap();
    detector.update(&features).unwrap();
    score
}

fn mean(scores: &[f64]) -> f64 {
    scores.iter().sum::<f64>() / scores.len() as f64
}

#[test]
fn account_takeover_burst_scores_above_normal_checkout_traffic() {
    let mut factory = CheckoutEventFactory::seeded(2026);
    let mut detector = detector();

    train_on_normal_checkouts(&mut detector, &mut factory);

    // Simulate a monitoring loop that scores each event against the current
    // forest, then commits it to the sliding window.
    let normal_scores: Vec<f64> = (0..64)
        .map(|_| score_then_update(&mut detector, factory.normal_checkout()))
        .collect();
    let attack_scores: Vec<f64> = (0..8)
        .map(|idx| score_then_update(&mut detector, factory.account_takeover_attempt(idx)))
        .collect();

    let mean_normal = mean(&normal_scores);
    let mean_attack = mean(&attack_scores);
    let attacks_above_normal_mean = attack_scores
        .iter()
        .filter(|&&score| score > mean_normal)
        .count();

    assert!(
        mean_attack > mean_normal,
        "attack burst should score higher on average: attack={mean_attack} normal={mean_normal}, attack_scores={attack_scores:?}, normal_scores={normal_scores:?}"
    );
    assert!(
        attacks_above_normal_mean >= 6,
        "most attack events should outrank the normal mean: count={attacks_above_normal_mean}, mean_normal={mean_normal}, attack_scores={attack_scores:?}, normal_scores={normal_scores:?}"
    );
}
