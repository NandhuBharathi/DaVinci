use davinci::learning::model::TrainingStepDiagnostics;
use davinci::learning::LearningModel;
use davinci::training::{
    SensorConfig, StabilityState, TrainingBrainController, TrainingDecision, TrainingSensors,
};

fn model() -> LearningModel {
    LearningModel::new(4, 3, 2).unwrap()
}

fn diagnostics() -> TrainingStepDiagnostics {
    TrainingStepDiagnostics {
        loss: 0.5,
        gradients: vec![0.1, 0.2, 0.3, 0.4],
        activations: vec![0.01, 0.02, 0.03],
    }
}

#[test]
fn weight_sensor_collects_statistics() {
    let snapshot = TrainingSensors::default()
        .inspect(&model(), &diagnostics(), 1, 1)
        .unwrap();

    assert!(snapshot.weights.count > 0);
    assert!(snapshot.weights.l2_norm.is_finite());
}

#[test]
fn gradient_sensor_collects_statistics() {
    let snapshot = TrainingSensors::default()
        .inspect(&model(), &diagnostics(), 1, 1)
        .unwrap();

    assert_eq!(snapshot.gradients.count, 4);
    assert!(snapshot.gradients.l2_norm > 0.0);
}

#[test]
fn activation_sensor_collects_values() {
    let snapshot = TrainingSensors::default()
        .inspect(&model(), &diagnostics(), 1, 1)
        .unwrap();

    assert_eq!(snapshot.activation_count, 3);
    assert!(snapshot.max_activation > 0.0);
}

#[test]
fn loss_sensor_records_loss() {
    let snapshot = TrainingSensors::default()
        .inspect(&model(), &diagnostics(), 5, 2)
        .unwrap();

    assert_eq!(snapshot.loss, 0.5);
    assert_eq!(snapshot.step, 5);
    assert_eq!(snapshot.epoch, 2);
}

#[test]
fn memory_sensor_estimates_model_memory() {
    let snapshot = TrainingSensors::default()
        .inspect(&model(), &diagnostics(), 1, 1)
        .unwrap();

    assert!(snapshot.estimated_memory_bytes > 0);
}

#[test]
fn normal_state_is_detected() {
    let snapshot = TrainingSensors::default()
        .inspect(&model(), &diagnostics(), 1, 1)
        .unwrap();

    assert_eq!(snapshot.stability, StabilityState::Normal);
}

#[test]
fn critical_gradient_is_detected() {
    let mut config = SensorConfig::default();
    config.warning_gradient_norm = 1.0;
    config.critical_gradient_norm = 2.0;

    let sensors = TrainingSensors::new(config).unwrap();

    let mut data = diagnostics();
    data.gradients = vec![10.0, 10.0];

    let snapshot = sensors.inspect(&model(), &data, 1, 1).unwrap();

    assert_eq!(snapshot.stability, StabilityState::Critical);
}

#[test]
fn brain_controller_selects_critical_decision() {
    let snapshot = TrainingSensors::default()
        .inspect(&model(), &diagnostics(), 1, 1)
        .unwrap();

    let controller = TrainingBrainController::new();

    assert_eq!(
        controller.decide(&snapshot, StabilityState::Critical),
        TrainingDecision::Critical
    );
}

#[test]
fn brain_controller_selects_warning_decision() {
    let snapshot = TrainingSensors::default()
        .inspect(&model(), &diagnostics(), 1, 1)
        .unwrap();

    let controller = TrainingBrainController::new();

    assert_eq!(
        controller.decide(&snapshot, StabilityState::Warning),
        TrainingDecision::Warning
    );
}

#[test]
fn brain_controller_selects_normal_decision() {
    let snapshot = TrainingSensors::default()
        .inspect(&model(), &diagnostics(), 1, 1)
        .unwrap();

    let controller = TrainingBrainController::new();

    assert_eq!(
        controller.decide(&snapshot, StabilityState::Normal),
        TrainingDecision::Normal
    );
}

#[test]
fn non_finite_activation_is_rejected() {
    let mut data = diagnostics();
    data.activations[0] = f32::NAN;

    assert!(TrainingSensors::default()
        .inspect(&model(), &data, 1, 1)
        .is_err());
}
#[test]
fn invalid_sensor_configuration_is_rejected() {
    let mut config = SensorConfig::default();
    config.critical_gradient_norm = 0.0;

    assert!(TrainingSensors::new(config).is_err());
}
