use davinci::learning::{GradientControlConfig, GradientController, LearningModel};

fn controller() -> GradientController {
    GradientController::default()
}

#[test]
fn normal_gradient_is_not_modified() {
    let values = vec![0.1, 0.2, 0.3];

    let (result, report) = controller().process(&values).unwrap();

    assert_eq!(result, values);
    assert_eq!(report.clipping_scale, 1.0);
    assert!(!report.exploded);
}

#[test]
fn large_gradient_is_clipped() {
    let values = vec![10.0, 10.0];

    let (result, report) = controller().process(&values).unwrap();

    assert!(report.raw.l2_norm > 5.0);
    assert!(report.clipped.l2_norm <= 5.0001);
    assert!(report.clipping_scale < 1.0);
    assert!(result.iter().all(|v| v.is_finite()));
}

#[test]
fn gradient_explosion_is_detected() {
    let values = vec![15.0, 15.0];

    let (_, report) = controller().process(&values).unwrap();

    assert!(report.exploded);
}

#[test]
fn non_finite_gradient_is_rejected() {
    let values = vec![0.1, f32::NAN];

    assert!(controller().process(&values).is_err());
}

#[test]
fn invalid_configuration_is_rejected() {
    let config = GradientControlConfig {
        max_norm: 0.0,
        ..GradientControlConfig::default()
    };

    assert!(GradientController::new(config).is_err());
}

#[test]
fn frozen_embeddings_are_not_updated() {
    let config = GradientControlConfig {
        freeze_embeddings: true,
        ..GradientControlConfig::default()
    };

    let controller = GradientController::new(config).unwrap();

    let mut model = LearningModel::new(3, 4, 2).unwrap();
    let before = model.embeddings.clone();

    let (_, report) = model
        .train_step_with_gradient_control(0, 1, 0.01, &controller)
        .unwrap();

    assert!(!report.exploded);
    assert_eq!(model.embeddings, before);
}

#[test]
fn frozen_output_weights_are_not_updated() {
    let config = GradientControlConfig {
        freeze_output_weights: true,
        ..GradientControlConfig::default()
    };

    let controller = GradientController::new(config).unwrap();

    let mut model = LearningModel::new(3, 4, 2).unwrap();
    let before = model.output_weights.clone();

    model
        .train_step_with_gradient_control(0, 1, 0.01, &controller)
        .unwrap();

    assert_eq!(model.output_weights, before);
}

#[test]
fn frozen_bias_is_not_updated() {
    let config = GradientControlConfig {
        freeze_output_bias: true,
        ..GradientControlConfig::default()
    };

    let controller = GradientController::new(config).unwrap();

    let mut model = LearningModel::new(3, 4, 2).unwrap();
    let before = model.output_bias.clone();

    model
        .train_step_with_gradient_control(0, 1, 0.01, &controller)
        .unwrap();

    assert_eq!(model.output_bias, before);
}

#[test]
fn all_parameters_can_be_frozen() {
    let config = GradientControlConfig {
        freeze_embeddings: true,
        freeze_output_weights: true,
        freeze_output_bias: true,
        ..GradientControlConfig::default()
    };

    let controller = GradientController::new(config).unwrap();

    let mut model = LearningModel::new(3, 4, 2).unwrap();

    let embeddings = model.embeddings.clone();
    let weights = model.output_weights.clone();
    let bias = model.output_bias.clone();

    model
        .train_step_with_gradient_control(0, 1, 0.01, &controller)
        .unwrap();

    assert_eq!(model.embeddings, embeddings);
    assert_eq!(model.output_weights, weights);
    assert_eq!(model.output_bias, bias);
}

#[test]
fn gradient_controlled_training_returns_diagnostics() {
    let mut model = LearningModel::new(3, 4, 2).unwrap();

    let (diagnostics, report) = model
        .train_step_with_gradient_control(0, 1, 0.01, &controller())
        .unwrap();

    assert!(diagnostics.loss.is_finite());
    assert!(!diagnostics.gradients.is_empty());
    assert!(report.raw.l2_norm.is_finite());
    assert!(report.clipped.l2_norm.is_finite());
}
