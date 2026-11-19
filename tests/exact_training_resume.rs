use davinci::learning::{CheckpointManager, LearningDataset, LearningPair, TrainingConfig};

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
    let base = format!("/tmp/davinci_phase12_{}_{}", name, std::process::id());

    let _ = std::fs::remove_dir_all(&base);

    let resume = ResumeManager::new(format!("{}/resume.bin", base));
    let checkpoints = CheckpointManager::new(format!("{}/checkpoints", base)).unwrap();

    ResumableTrainer {
        resume,
        checkpoints,
        monitor: TrainingMonitor::default(),
    }
}

#[test]
fn training_can_pause_at_exact_step() {
    let trainer = trainer("pause");

    let (_, report) = trainer.train(&dataset(), &config(), Some(2)).unwrap();

    assert_eq!(report.status, ResumeStatus::Paused);
    assert_eq!(report.steps, 2);
    assert!(trainer.resume.exists());
}

#[test]
fn training_resumes_from_saved_step() {
    let trainer = trainer("resume");

    let data = dataset();
    let config = config();

    let (_, first) = trainer.train(&data, &config, Some(2)).unwrap();

    assert_eq!(first.steps, 2);

    let (_, second) = trainer.train(&data, &config, Some(2)).unwrap();

    assert!(second.resumed);
    assert_eq!(second.steps, 4);
}

#[test]
fn resumed_training_completes() {
    let trainer = trainer("complete");

    let data = dataset();
    let config = config();

    let (_, first) = trainer.train(&data, &config, Some(2)).unwrap();

    assert_eq!(first.status, ResumeStatus::Paused);

    let (_, second) = trainer.train(&data, &config, None).unwrap();

    assert_eq!(second.status, ResumeStatus::Completed);
    assert_eq!(second.steps, 12);
    assert!(!trainer.resume.exists());
}

#[test]
fn dataset_changes_are_rejected() {
    let trainer = trainer("dataset");

    trainer.train(&dataset(), &config(), Some(2)).unwrap();

    let changed = LearningDataset {
        pairs: vec![LearningPair {
            input: "changed".into(),
            target: "different".into(),
        }],
    };

    assert!(trainer.train(&changed, &config(), None,).is_err());
}

#[test]
fn training_configuration_changes_are_rejected() {
    let trainer = trainer("config");

    trainer.train(&dataset(), &config(), Some(2)).unwrap();

    let changed = TrainingConfig {
        epochs: 4,
        learning_rate: 0.2,
        hidden_size: 8,
    };

    assert!(trainer.train(&dataset(), &changed, None,).is_err());
}

#[test]
fn resume_checkpoint_corruption_is_detected() {
    let trainer = trainer("corrupt");

    trainer.train(&dataset(), &config(), Some(2)).unwrap();

    let path = trainer.resume.path();

    let mut bytes = std::fs::read(path).unwrap();

    let last = bytes.len() - 1;
    bytes[last] ^= 0xFF;

    std::fs::write(path, bytes).unwrap();

    assert!(trainer.train(&dataset(), &config(), None,).is_err());
}

#[test]
fn uninterrupted_and_resumed_training_match() {
    let uninterrupted = trainer("uninterrupted");
    let resumed = trainer("split");

    let data = dataset();
    let config = config();

    let (full_model, _) = uninterrupted.train(&data, &config, None).unwrap();

    resumed.train(&data, &config, Some(3)).unwrap();

    let (resumed_model, report) = resumed.train(&data, &config, None).unwrap();

    assert_eq!(report.status, ResumeStatus::Completed);
    assert_eq!(full_model.embeddings, resumed_model.embeddings);
    assert_eq!(full_model.output_weights, resumed_model.output_weights);
    assert_eq!(full_model.output_bias, resumed_model.output_bias);
}
