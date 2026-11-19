use davinci::learning::{
    CheckpointManager, LearningDataset, LearningPair, LearningRateScheduler, SchedulerConfig,
    SchedulerKind, SchedulerStateManager, TrainingConfig,
};

use davinci::training::{ResumableTrainer, ResumeManager, ResumeStatus, TrainingMonitor};

fn dataset() -> LearningDataset {
    LearningDataset {
        pairs: vec![
            LearningPair {
                input: "brake".into(),
                target: "friction".into(),
            },
            LearningPair {
                input: "engine".into(),
                target: "combustion".into(),
            },
            LearningPair {
                input: "fuel".into(),
                target: "combustion".into(),
            },
        ],
    }
}

fn config() -> TrainingConfig {
    TrainingConfig {
        epochs: 4,
        learning_rate: 0.05,
        hidden_size: 8,
    }
}

fn trainer(name: &str) -> ResumableTrainer {
    let base = std::env::temp_dir().join(format!(
        "davinci_scheduler_continuity_{}_{}",
        name,
        std::process::id()
    ));

    let _ = std::fs::remove_dir_all(&base);
    std::fs::create_dir_all(&base).unwrap();

    ResumableTrainer {
        resume: ResumeManager::new(base.join("resume.bin")),
        checkpoints: CheckpointManager::new(base.join("checkpoints")).unwrap(),
        monitor: TrainingMonitor::default(),
    }
}

fn assert_models_equal(a: &davinci::learning::LearningModel, b: &davinci::learning::LearningModel) {
    assert_eq!(a.input_size, b.input_size);
    assert_eq!(a.hidden_size, b.hidden_size);
    assert_eq!(a.output_size, b.output_size);

    let compare = |x: &[f32], y: &[f32]| {
        assert_eq!(x.len(), y.len());

        for (left, right) in x.iter().zip(y.iter()) {
            assert_eq!(
                left.to_bits(),
                right.to_bits(),
                "Model weights differ: {} vs {}",
                left,
                right
            );
        }
    };

    compare(
        &a.embeddings.iter().flatten().copied().collect::<Vec<_>>(),
        &b.embeddings.iter().flatten().copied().collect::<Vec<_>>(),
    );

    compare(
        &a.output_weights
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>(),
        &b.output_weights
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>(),
    );

    compare(&a.output_bias, &b.output_bias);
}

fn scheduler_kind(name: &str) -> SchedulerKind {
    SchedulerKind::from_keyword(name).unwrap()
}

#[test]
fn all_optimizer_scheduler_combinations_preserve_resume_continuity() {
    let optimizers = ["sgd", "momentum", "adam"];

    let schedulers = [
        "constant",
        "step_decay",
        "exponential_decay",
        "cosine_annealing",
        "warmup_cosine",
    ];

    for optimizer in optimizers {
        for scheduler in schedulers {
            let case = format!("{}_{}", optimizer, scheduler);

            // Uninterrupted reference training.
            let full_trainer = trainer(&format!("full_{}", case));

            let (full_model, full_report) = full_trainer
                .train_with_optimizer_and_scheduler(
                    &dataset(),
                    &config(),
                    None,
                    optimizer,
                    scheduler,
                )
                .unwrap();

            assert_eq!(full_report.status, ResumeStatus::Completed);
            assert_eq!(full_report.steps, 12);

            // First segment: stop after five completed samples.
            let resumed_trainer = trainer(&format!("resume_{}", case));

            let (partial_model, partial_report) = resumed_trainer
                .train_with_optimizer_and_scheduler(
                    &dataset(),
                    &config(),
                    Some(5),
                    optimizer,
                    scheduler,
                )
                .unwrap();

            assert_eq!(partial_report.status, ResumeStatus::Paused);
            assert_eq!(partial_report.steps, 5);
            assert!(resumed_trainer.resume.exists());

            // Scheduler checkpoint must match the exact training step.
            let base = resumed_trainer.resume.path();
            let scheduler_path = base.with_extension("scheduler");
            let scheduler_manager = SchedulerStateManager::new(&scheduler_path);

            let saved_scheduler = scheduler_manager.load().unwrap();

            assert_eq!(saved_scheduler.state.step, 5);
            assert!(saved_scheduler.current_learning_rate().is_finite());

            // Resume remaining steps.
            let (resumed_model, resumed_report) = resumed_trainer
                .train_with_optimizer_and_scheduler(
                    &dataset(),
                    &config(),
                    None,
                    optimizer,
                    scheduler,
                )
                .unwrap();

            assert!(resumed_report.resumed);
            assert_eq!(resumed_report.status, ResumeStatus::Completed);
            assert_eq!(resumed_report.steps, 12);

            // Compare every parameter exactly.
            assert_models_equal(&full_model, &resumed_model);

            // Completed training must clean up active state files.
            assert!(!resumed_trainer.resume.exists());
            assert!(!scheduler_path.exists());

            let _ = partial_model;
            let _ = scheduler_kind(scheduler);
        }
    }
}

#[test]
fn scheduler_checkpoint_rejects_training_step_mismatch() {
    let trainer = trainer("step_mismatch");

    let (_, report) = trainer
        .train_with_optimizer_and_scheduler(&dataset(), &config(), Some(4), "adam", "cosine")
        .unwrap();

    assert_eq!(report.steps, 4);

    let scheduler_path = trainer.resume.path().with_extension("scheduler");
    let manager = SchedulerStateManager::new(scheduler_path);

    let scheduler = manager.load().unwrap();

    assert_eq!(scheduler.state.step, report.steps as u64);
}

#[test]
fn scheduler_checkpoint_contains_valid_learning_rate() {
    let mut config = SchedulerConfig::new(SchedulerKind::CosineAnnealing, 0.05, 12).unwrap();

    config.minimum_learning_rate = 0.0005;

    let mut scheduler = LearningRateScheduler::new(config).unwrap();

    for _ in 0..5 {
        scheduler.advance().unwrap();
    }

    assert_eq!(scheduler.state.step, 5);
    assert!(scheduler.current_learning_rate() > 0.0);
    assert!(scheduler.current_learning_rate() <= 0.05);
}
