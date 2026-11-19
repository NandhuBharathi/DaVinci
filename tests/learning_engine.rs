use davinci::learning::{
    LearningDataset, LearningModel, LearningPair, LearningTrainer, TrainingConfig,
};

#[test]
fn model_initialization() {
    let model = LearningModel::new(5, 8, 3).unwrap();

    assert_eq!(model.input_size, 5);
    assert_eq!(model.hidden_size, 8);
    assert_eq!(model.output_size, 3);
}

#[test]
fn probability_distribution_is_valid() {
    let model = LearningModel::new(2, 8, 3).unwrap();
    let probabilities = model.probabilities(0).unwrap();

    let total: f32 = probabilities.iter().sum();

    assert!((total - 1.0).abs() < 0.0001);
}

#[test]
fn training_changes_weights() {
    let mut model = LearningModel::new(2, 8, 2).unwrap();

    let before = model.embeddings.clone();

    model.train_step(0, 1, 0.1).unwrap();

    assert_ne!(before, model.embeddings);
}

#[test]
fn training_reduces_loss_on_repeated_pair() {
    let mut model = LearningModel::new(1, 8, 2).unwrap();

    let before = model.probabilities(0).unwrap()[0];

    for _ in 0..100 {
        model.train_step(0, 0, 0.1).unwrap();
    }

    let after = model.probabilities(0).unwrap()[0];

    assert!(after > before);
}

#[test]
fn dataset_vocabulary_is_created() {
    let dataset = LearningDataset {
        pairs: vec![
            LearningPair {
                input: "brake_disc".into(),
                target: "braking_component".into(),
            },
            LearningPair {
                input: "friction".into(),
                target: "surface_interaction".into(),
            },
        ],
    };

    let (inputs, targets) = dataset.vocabulary();

    assert_eq!(inputs.len(), 2);
    assert_eq!(targets.len(), 2);
}

#[test]
fn training_engine_updates_model() {
    let dataset = LearningDataset {
        pairs: vec![
            LearningPair {
                input: "brake_disc".into(),
                target: "braking_component".into(),
            },
            LearningPair {
                input: "friction".into(),
                target: "surface_interaction".into(),
            },
        ],
    };

    let (_, report) = LearningTrainer::train(
        &dataset,
        &TrainingConfig {
            epochs: 20,
            learning_rate: 0.1,
            hidden_size: 8,
        },
    )
    .unwrap();

    assert!(report.weights_changed);
    assert!(report.steps > 0);
}

#[test]
fn invalid_model_dimensions_are_rejected() {
    assert!(LearningModel::new(0, 8, 2).is_err());
}

#[test]
fn invalid_learning_rate_is_rejected() {
    let mut model = LearningModel::new(1, 8, 2).unwrap();

    assert!(model.train_step(0, 0, 0.0).is_err());
}

#[test]
fn invalid_training_configuration_is_rejected() {
    let dataset = LearningDataset {
        pairs: vec![LearningPair {
            input: "x".into(),
            target: "y".into(),
        }],
    };

    let result = LearningTrainer::train(
        &dataset,
        &TrainingConfig {
            epochs: 0,
            learning_rate: 0.1,
            hidden_size: 8,
        },
    );

    assert!(result.is_err());
}

#[test]
fn dataset_file_is_loaded() {
    let dataset = LearningDataset::load("data/training_guidance.dv").unwrap();

    assert!(!dataset.pairs.is_empty());
}
