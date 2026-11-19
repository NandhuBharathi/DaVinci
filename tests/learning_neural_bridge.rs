use davinci::brain::NeuralConfig;
use davinci::core::DeviceConfig;
use davinci::learning::{LearningDataset, LearningPair, LearningTrainer, TrainingConfig};
use davinci::runtime::ActualBrainProcessor;

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
                input: "brake".into(),
                target: "friction".into(),
            },
            LearningPair {
                input: "engine".into(),
                target: "combustion".into(),
            },
        ],
    }
}

fn config() -> TrainingConfig {
    TrainingConfig {
        epochs: 100,
        learning_rate: 0.1,
        hidden_size: 8,
    }
}

fn processor() -> ActualBrainProcessor {
    ActualBrainProcessor::new(
        NeuralConfig {
            input_size: 32,
            hidden_size: 16,
            output_size: 8,
        },
        DeviceConfig::CPU,
    )
    .unwrap()
}

#[test]
fn training_creates_weight_updates() {
    let data = dataset();

    let (_, report) = LearningTrainer::train(&data, &config()).unwrap();

    assert!(report.steps > 0);
    assert!(report.weights_changed);
}

#[test]
fn trained_model_is_installed_into_neural_engine() {
    let mut brain = processor();

    assert!(!brain.brain().has_learned_model());

    brain.train_and_install(&dataset(), &config()).unwrap();

    assert!(brain.brain().has_learned_model());
}

#[test]
fn trained_model_generates_prediction() {
    let mut brain = processor();

    brain.train_and_install(&dataset(), &config()).unwrap();

    let prediction = brain.brain().predict_learned("brake").unwrap();

    assert!(prediction.is_some());
}

#[test]
fn trained_model_learns_repeated_pairs() {
    let mut brain = processor();

    brain.train_and_install(&dataset(), &config()).unwrap();

    let prediction = brain.brain().predict_learned("brake").unwrap();

    assert_eq!(prediction.as_deref(), Some("friction"));
}

#[test]
fn unknown_words_do_not_generate_fake_prediction() {
    let mut brain = processor();

    brain.train_and_install(&dataset(), &config()).unwrap();

    let prediction = brain.brain().predict_learned("unknownword").unwrap();

    assert!(prediction.is_none());
}

#[test]
fn empty_dataset_is_rejected() {
    let mut brain = processor();

    let empty = LearningDataset { pairs: Vec::new() };

    assert!(brain.train_and_install(&empty, &config()).is_err());
}

#[test]
fn inference_uses_installed_learned_model() {
    use davinci::runtime::{BrainInput, BrainProcessor, ProcessingMode};

    let mut brain = processor();

    brain.train_and_install(&dataset(), &config()).unwrap();

    let input = BrainInput {
        text: "brake".into(),
        context: Vec::new(),
        knowledge: Vec::new(),
        mode: ProcessingMode::Inference,
    };

    let output = brain.process(&input).unwrap();

    assert!(output.response.contains("friction"));
}

#[test]
fn trained_weights_are_not_changed_by_prediction() {
    let mut brain = processor();

    brain.train_and_install(&dataset(), &config()).unwrap();

    let before = brain.brain().predict_learned("brake").unwrap();

    for _ in 0..10 {
        brain.brain().predict_learned("brake").unwrap();
    }

    let after = brain.brain().predict_learned("brake").unwrap();

    assert_eq!(before, after);
}
