use davinci::brain::{IntegratedBrain, Intent, NeuralConfig, NeuralEngine, RuleEngine};

use davinci::core::{DeviceConfig, ProcessingMode, ProcessorKind};

fn neural_config() -> NeuralConfig {
    NeuralConfig {
        input_size: 64,
        hidden_size: 32,
        output_size: 16,
    }
}

#[test]
fn definition_question_detection() {
    let engine = RuleEngine::new();

    let result = engine.analyze("What is AI?");

    assert_eq!(result.intent, Intent::DefinitionQuestion);

    assert_eq!(result.concept, Some("ai".to_string()));
}

#[test]
fn question_detection() {
    let engine = RuleEngine::new();

    let result = engine.analyze("Who are you?");

    assert_eq!(result.intent, Intent::GeneralQuestion);
}

#[test]
fn statement_detection() {
    let engine = RuleEngine::new();

    let result = engine.analyze("AI is technology");

    assert_eq!(result.intent, Intent::Statement);
}

#[test]
fn neural_initialization() {
    let engine = NeuralEngine::new(neural_config());

    assert!(engine.is_ok());
}

#[test]
fn neural_forward_computation() {
    let engine = NeuralEngine::new(neural_config()).unwrap();

    let input = vec![0.5; 64];

    let output = engine.forward(&input).unwrap();

    assert_eq!(output.len(), 16);
}

#[test]
fn neural_invalid_input() {
    let engine = NeuralEngine::new(neural_config()).unwrap();

    let input = vec![0.5; 10];

    assert!(engine.forward(&input).is_err());
}

#[test]
fn integrated_brain_processing() {
    let brain = IntegratedBrain::new(neural_config(), DeviceConfig::CPU).unwrap();

    let result = brain
        .process("What is AI?", ProcessingMode::Inference)
        .unwrap();

    assert_eq!(result.rule_analysis.intent, Intent::DefinitionQuestion);

    assert_eq!(result.neural_representation.len(), 16);
}

#[test]
fn integrated_brain_processors() {
    let brain = IntegratedBrain::new(neural_config(), DeviceConfig::CPU).unwrap();

    let result = brain
        .process("What is AI?", ProcessingMode::Training)
        .unwrap();

    assert!(result.context.has_processor(ProcessorKind::RuleBased));

    assert!(result.context.has_processor(ProcessorKind::Neural));
}

#[test]
fn invalid_neural_configuration() {
    let config = NeuralConfig {
        input_size: 0,
        hidden_size: 32,
        output_size: 16,
    };

    assert!(NeuralEngine::new(config).is_err());
}
