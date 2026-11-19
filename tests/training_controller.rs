use davinci::learning::{CheckpointManager, LearningDataset, LearningPair, TrainingConfig};

use davinci::training::{
    ControllerConfig, MonitorConfig, StabilityState, TrainingController, TrainingMonitor,
    TrainingStatus,
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

fn training_config() -> TrainingConfig {
    TrainingConfig {
        epochs: 3,
        learning_rate: 0.05,
        hidden_size: 8,
    }
}

fn controller(name: &str, max_steps: Option<usize>) -> TrainingController {
    let path = format!("/tmp/davinci_phase11_{}_{}", name, std::process::id());

    let _ = std::fs::remove_dir_all(&path);

    let checkpoints = CheckpointManager::new(path).unwrap();

    TrainingController::new(
        checkpoints,
        ControllerConfig {
            max_steps,
            monitor: TrainingMonitor::default(),
        },
    )
}

#[test]
fn training_completes_all_epochs() {
    let controller = controller("complete", None);

    let (_, report) = controller.train(&dataset(), &training_config()).unwrap();

    assert_eq!(report.status, TrainingStatus::Completed);
    assert_eq!(report.completed_epochs, 3);
    assert_eq!(report.steps, 9);
}

#[test]
fn training_stops_at_step_limit() {
    let controller = controller("pause", Some(4));

    let (_, report) = controller.train(&dataset(), &training_config()).unwrap();

    assert_eq!(report.status, TrainingStatus::PausedAtStepLimit);
    assert_eq!(report.steps, 4);
}

#[test]
fn latest_checkpoint_is_created() {
    let controller = controller("latest", None);

    controller.train(&dataset(), &training_config()).unwrap();

    assert!(controller.checkpoints.latest_path().exists());
}

#[test]
fn best_checkpoint_is_created() {
    let controller = controller("best", None);

    controller.train(&dataset(), &training_config()).unwrap();

    assert!(controller.checkpoints.best_path().exists());
}

#[test]
fn training_metrics_are_valid() {
    let model = davinci::learning::LearningModel::new(3, 4, 2).unwrap();

    let monitor = TrainingMonitor::default();

    let (metrics, state) = monitor.inspect(&model, 0.5, 1, 1).unwrap();

    assert_eq!(metrics.step, 1);
    assert_eq!(state, StabilityState::Normal);
}

#[test]
fn warning_threshold_is_detected() {
    let monitor = TrainingMonitor::new(MonitorConfig {
        warning_loss: 1.0,
        critical_loss: 10.0,
        max_weight_norm: 1000.0,
        max_absolute_weight: 100.0,
    })
    .unwrap();

    let model = davinci::learning::LearningModel::new(3, 4, 2).unwrap();

    let (_, state) = monitor.inspect(&model, 2.0, 1, 1).unwrap();

    assert_eq!(state, StabilityState::Warning);
}

#[test]
fn critical_threshold_is_detected() {
    let monitor = TrainingMonitor::new(MonitorConfig {
        warning_loss: 1.0,
        critical_loss: 5.0,
        max_weight_norm: 1000.0,
        max_absolute_weight: 100.0,
    })
    .unwrap();

    let model = davinci::learning::LearningModel::new(3, 4, 2).unwrap();

    let (_, state) = monitor.inspect(&model, 6.0, 1, 1).unwrap();

    assert_eq!(state, StabilityState::Critical);
}

#[test]
fn non_finite_loss_is_rejected() {
    let model = davinci::learning::LearningModel::new(3, 4, 2).unwrap();

    assert!(TrainingMonitor::default()
        .inspect(&model, f32::NAN, 1, 1)
        .is_err());
}

#[test]
fn empty_dataset_is_rejected() {
    let controller = controller("empty", None);

    let empty = LearningDataset { pairs: Vec::new() };

    assert!(controller.train(&empty, &training_config()).is_err());
}
