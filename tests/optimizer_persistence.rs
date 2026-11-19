use davinci::learning::{
    GradientController, LearningModel, OptimizerConfig, OptimizerEngine, OptimizerKind,
    OptimizerStateManager,
};

fn create_optimizer(kind: OptimizerKind) -> OptimizerEngine {
    OptimizerEngine::new(OptimizerConfig::new(kind, 0.01).unwrap()).unwrap()
}

fn train_steps(
    model: &mut LearningModel,
    optimizer: &mut OptimizerEngine,
    start: usize,
    end: usize,
) {
    let controller = GradientController::default();

    let sequence = [(0usize, 1usize), (1, 2), (2, 0), (0, 2), (1, 0), (2, 1)];

    for step in start..end {
        let (input, target) = sequence[step % sequence.len()];

        let (diagnostics, report) = model
            .calculate_gradients(input, target, &controller)
            .unwrap();

        assert!(!report.exploded);

        let config = controller.config();

        optimizer
            .update(
                model,
                input,
                &diagnostics.gradients,
                config.freeze_embeddings,
                config.freeze_output_weights,
                config.freeze_output_bias,
            )
            .unwrap();
    }
}

fn assert_models_equal(a: &LearningModel, b: &LearningModel) {
    assert_eq!(a.embeddings, b.embeddings);
    assert_eq!(a.output_weights, b.output_weights);
    assert_eq!(a.output_bias, b.output_bias);
}

fn run_continuity_test(kind: OptimizerKind) {
    let directory = std::env::temp_dir().join(format!(
        "davinci_optimizer_test_{}_{}",
        std::process::id(),
        kind.keyword()
    ));

    std::fs::create_dir_all(&directory).unwrap();

    let path = directory.join("optimizer.state");
    let manager = OptimizerStateManager::new(&path);

    let mut uninterrupted_model = LearningModel::new(3, 4, 3).unwrap();

    let mut uninterrupted_optimizer = create_optimizer(kind);

    train_steps(
        &mut uninterrupted_model,
        &mut uninterrupted_optimizer,
        0,
        12,
    );

    let mut interrupted_model = LearningModel::new(3, 4, 3).unwrap();

    let mut interrupted_optimizer = create_optimizer(kind);

    train_steps(&mut interrupted_model, &mut interrupted_optimizer, 0, 5);

    manager
        .save(&interrupted_optimizer, &interrupted_model, 5)
        .unwrap();

    let mut restored_optimizer = manager.load(&interrupted_model, 5).unwrap();

    train_steps(&mut interrupted_model, &mut restored_optimizer, 5, 12);

    assert_models_equal(&uninterrupted_model, &interrupted_model);

    assert_eq!(
        uninterrupted_optimizer.state.step,
        restored_optimizer.state.step,
    );

    assert_eq!(
        uninterrupted_optimizer.state.first_moment,
        restored_optimizer.state.first_moment,
    );

    assert_eq!(
        uninterrupted_optimizer.state.second_moment,
        restored_optimizer.state.second_moment,
    );

    manager.remove().unwrap();
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn sgd_training_continuity() {
    run_continuity_test(OptimizerKind::Sgd);
}

#[test]
fn momentum_training_continuity() {
    run_continuity_test(OptimizerKind::Momentum);
}

#[test]
fn adam_training_continuity() {
    run_continuity_test(OptimizerKind::Adam);
}

#[test]
fn optimizer_rejects_wrong_resume_step() {
    let directory = std::env::temp_dir().join(format!(
        "davinci_optimizer_step_test_{}",
        std::process::id()
    ));

    std::fs::create_dir_all(&directory).unwrap();

    let path = directory.join("optimizer.state");
    let manager = OptimizerStateManager::new(&path);

    let mut model = LearningModel::new(3, 4, 3).unwrap();
    let mut optimizer = create_optimizer(OptimizerKind::Adam);

    train_steps(&mut model, &mut optimizer, 0, 3);

    manager.save(&optimizer, &model, 3).unwrap();

    assert!(manager.load(&model, 2).is_err());

    manager.remove().unwrap();
    std::fs::remove_dir_all(directory).unwrap();
}
