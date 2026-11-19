use davinci::learning::{
    LearningRateScheduler, SchedulerConfig, SchedulerKind, SchedulerStateManager,
};

fn create(kind: SchedulerKind) -> LearningRateScheduler {
    let mut config = SchedulerConfig::new(kind, 0.01, 100).unwrap();

    config.minimum_learning_rate = 0.001;
    config.step_size = 10;
    config.decay_factor = 0.5;
    config.warmup_steps = 10;

    LearningRateScheduler::new(config).unwrap()
}

#[test]
fn constant_scheduler_keeps_rate() {
    let mut scheduler = create(SchedulerKind::Constant);

    for _ in 0..20 {
        assert!((scheduler.advance().unwrap() - 0.01).abs() < 1e-6);
    }
}

#[test]
fn step_decay_reduces_rate() {
    let mut scheduler = create(SchedulerKind::StepDecay);

    for _ in 0..10 {
        scheduler.advance().unwrap();
    }

    assert!((scheduler.current_learning_rate() - 0.005).abs() < 1e-6);
}

#[test]
fn exponential_decay_reduces_rate() {
    let mut scheduler = create(SchedulerKind::ExponentialDecay);

    let initial = scheduler.current_learning_rate();

    for _ in 0..10 {
        scheduler.advance().unwrap();
    }

    assert!(scheduler.current_learning_rate() < initial);
}

#[test]
fn cosine_reaches_minimum_at_end() {
    let mut scheduler = create(SchedulerKind::CosineAnnealing);

    for _ in 0..100 {
        scheduler.advance().unwrap();
    }

    assert!((scheduler.current_learning_rate() - 0.001).abs() < 1e-6);
}

#[test]
fn warmup_increases_rate() {
    let mut scheduler = create(SchedulerKind::WarmupCosine);

    let first = scheduler.advance().unwrap();
    let second = scheduler.advance().unwrap();

    assert!(second > first);
}

#[test]
fn scheduler_state_persistence() {
    let directory =
        std::env::temp_dir().join(format!("davinci_scheduler_test_{}", std::process::id()));

    std::fs::create_dir_all(&directory).unwrap();

    let path = directory.join("scheduler.state");
    let manager = SchedulerStateManager::new(&path);

    let mut original = create(SchedulerKind::WarmupCosine);

    for _ in 0..37 {
        original.advance().unwrap();
    }

    manager.save(&original).unwrap();

    let restored = manager.load().unwrap();

    assert_eq!(restored.state.step, original.state.step);

    assert_eq!(
        restored.current_learning_rate().to_bits(),
        original.current_learning_rate().to_bits()
    );

    std::fs::remove_file(path).unwrap();
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn scheduler_rejects_invalid_keyword() {
    assert!(SchedulerKind::from_keyword("unknown").is_err());
}
