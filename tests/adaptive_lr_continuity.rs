use davinci::learning::{
    CheckpointManager, LearningDataset, LearningModel, LearningPair, SchedulerStateManager,
    TrainingConfig,
};

use davinci::training::{
    MonitorConfig, ResumableTrainer, ResumeManager, ResumeStatus, TrainingMonitor,
};

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

// Every valid loss triggers Warning.
// Critical threshold remains safely above normal loss.
fn warning_monitor() -> TrainingMonitor {
    TrainingMonitor::new(MonitorConfig {
        warning_loss: 0.0,
        critical_loss: 20.0,
        max_weight_norm: 1000.0,
        max_absolute_weight: 100.0,
    })
    .unwrap()
}

fn trainer(name: &str) -> ResumableTrainer {
    let base =
        std::env::temp_dir().join(format!("davinci_adaptive_{}_{}", name, std::process::id()));

    let _ = std::fs::remove_dir_all(&base);
    std::fs::create_dir_all(&base).unwrap();

    ResumableTrainer {
        resume: ResumeManager::new(base.join("resume.bin")),
        checkpoints: CheckpointManager::new(base.join("checkpoints")).unwrap(),
        monitor: warning_monitor(),
    }
}

fn adaptive_scale(trainer: &ResumableTrainer) -> f32 {
    let path = trainer.resume.path().with_extension("adaptive");
    let bytes = std::fs::read(path).unwrap();

    assert_eq!(bytes.len(), 20);
    assert_eq!(&bytes[..8], b"DAVADP18");

    let expected = u64::from_le_bytes(bytes[12..20].try_into().unwrap());

    let mut hash = 14695981039346656037u64;
    for byte in &bytes[..12] {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(1099511628211);
    }

    assert_eq!(hash, expected, "Adaptive checkpoint checksum mismatch");

    f32::from_le_bytes(bytes[8..12].try_into().unwrap())
}

fn assert_models_equal(a: &LearningModel, b: &LearningModel) {
    assert_eq!(a.input_size, b.input_size);
    assert_eq!(a.hidden_size, b.hidden_size);
    assert_eq!(a.output_size, b.output_size);

    fn compare(a: &[f32], b: &[f32]) {
        assert_eq!(a.len(), b.len());

        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.to_bits(), y.to_bits(), "Weight mismatch: {x} vs {y}");
        }
    }

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

#[test]
fn adaptive_lr_persists_across_all_optimizer_scheduler_combinations() {
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
            let case = format!("{optimizer}_{scheduler}");

            // Full uninterrupted training.
            let full = trainer(&format!("full_{case}"));

            let (full_model, full_report) = full
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

            // Interrupted training.
            let split = trainer(&format!("split_{case}"));

            let (_, partial_report) = split
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
            assert!(split.resume.exists());

            // Five warnings should reduce scale to its 0.1 floor.
            let saved_scale = adaptive_scale(&split);

            assert!(
                (saved_scale - 0.1).abs() < 1e-6,
                "Unexpected adaptive scale for {case}: {saved_scale}"
            );

            // Scheduler and training step must remain synchronized.
            let scheduler_manager =
                SchedulerStateManager::new(split.resume.path().with_extension("scheduler"));

            let saved_scheduler = scheduler_manager.load().unwrap();

            assert_eq!(saved_scheduler.state.step, 5);

            // Resume remaining training.
            let (resumed_model, resumed_report) = split
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

            // Exact bit-level comparison.
            assert_models_equal(&full_model, &resumed_model);

            // Active state files must be cleaned after completion.
            assert!(!split.resume.exists());
            assert!(!split.resume.path().with_extension("adaptive").exists());
            assert!(!split.resume.path().with_extension("scheduler").exists());
            assert!(!split.resume.path().with_extension("optimizer").exists());
        }
    }
}
