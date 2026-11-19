use davinci::training::{TrainingBrainController, TrainingDecision};

#[test]
fn normal_preserves_learning_rate() {
    let c = TrainingBrainController::new();
    assert_eq!(
        c.adjust_learning_rate(0.05, 0.05, TrainingDecision::Normal)
            .unwrap(),
        0.05
    );
}

#[test]
fn warning_halves_learning_rate() {
    let c = TrainingBrainController::new();
    let lr = c
        .adjust_learning_rate(0.05, 0.05, TrainingDecision::Warning)
        .unwrap();
    assert!((lr - 0.025).abs() < 1e-7);
}

#[test]
fn warning_respects_minimum_rate() {
    let c = TrainingBrainController::new();
    let lr = c
        .adjust_learning_rate(0.05, 0.005, TrainingDecision::Warning)
        .unwrap();
    assert!((lr - 0.005).abs() < 1e-7);
}

#[test]
fn critical_returns_zero() {
    let c = TrainingBrainController::new();
    assert_eq!(
        c.adjust_learning_rate(0.05, 0.025, TrainingDecision::Critical)
            .unwrap(),
        0.0
    );
}

#[test]
fn invalid_rate_is_rejected() {
    let c = TrainingBrainController::new();
    assert!(c
        .adjust_learning_rate(f32::NAN, 0.05, TrainingDecision::Warning)
        .is_err());
}
