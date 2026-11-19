use davinci::learning::{
    CheckpointManager, GradientMonitor, LearningModel, WeightMonitor, WeightRegistry,
};

use std::fs;

fn model() -> LearningModel {
    LearningModel::new(3, 4, 2).unwrap()
}

fn vocab() -> (Vec<String>, Vec<String>) {
    (
        vec!["brake".into(), "engine".into(), "fuel".into()],
        vec!["friction".into(), "combustion".into()],
    )
}

fn manager(name: &str) -> CheckpointManager {
    let path = format!("/tmp/davinci_phase10_{}_{}", name, std::process::id());

    let _ = fs::remove_dir_all(&path);

    CheckpointManager::new(path).unwrap()
}

#[test]
fn weight_registry_collects_statistics() {
    let stats = WeightRegistry::collect(&model()).unwrap();

    assert!(stats.count > 0);
    assert!(stats.min <= stats.max);
    assert!(stats.l2_norm.is_finite());
}

#[test]
fn weight_monitor_validates_model() {
    assert!(WeightMonitor::validate(&model()).is_ok());
}

#[test]
fn gradient_monitor_detects_non_finite_values() {
    assert!(GradientMonitor::inspect(&[1.0, f32::NAN]).is_err());
}

#[test]
fn gradient_clipping_limits_norm() {
    let mut gradients = vec![3.0, 4.0];

    GradientMonitor::clip(&mut gradients, 2.0).unwrap();

    let norm = gradients
        .iter()
        .map(|v| (*v as f64).powi(2))
        .sum::<f64>()
        .sqrt();

    assert!((norm - 2.0).abs() < 0.0001);
}

#[test]
fn latest_checkpoint_round_trip() {
    let manager = manager("latest");
    let model = model();
    let (inputs, targets) = vocab();

    manager.save_latest(&model, &inputs, &targets, 0.5).unwrap();

    let loaded = manager.load_latest().unwrap();

    assert_eq!(loaded.model.embeddings, model.embeddings);
    assert_eq!(loaded.model.output_weights, model.output_weights);
    assert_eq!(loaded.input_vocabulary, inputs);
    assert_eq!(loaded.target_vocabulary, targets);
    assert_eq!(loaded.validation_loss, 0.5);
}

#[test]
fn best_checkpoint_only_improves() {
    let manager = manager("best");
    let model = model();
    let (inputs, targets) = vocab();

    assert!(manager
        .save_best_if_improved(&model, &inputs, &targets, 0.8)
        .unwrap());

    assert!(!manager
        .save_best_if_improved(&model, &inputs, &targets, 0.9)
        .unwrap());

    assert!(manager
        .save_best_if_improved(&model, &inputs, &targets, 0.3)
        .unwrap());

    assert_eq!(manager.load_best().unwrap().validation_loss, 0.3);
}

#[test]
fn checkpoint_rejects_invalid_loss() {
    let manager = manager("invalid");
    let model = model();
    let (inputs, targets) = vocab();

    assert!(manager
        .save_latest(&model, &inputs, &targets, f32::NAN)
        .is_err());
}

#[test]
fn corrupted_checkpoint_is_rejected() {
    let manager = manager("corrupt");
    let model = model();
    let (inputs, targets) = vocab();

    manager.save_latest(&model, &inputs, &targets, 0.5).unwrap();

    let path = manager.latest_path();

    let mut bytes = fs::read(&path).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 0xFF;

    fs::write(path, bytes).unwrap();

    assert!(manager.load_latest().is_err());
}
