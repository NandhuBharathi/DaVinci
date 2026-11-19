use davinci::brain::NeuralConfig;
use davinci::core::DeviceConfig;
use davinci::memory::MemoryController;
use davinci::runtime::{
    ActualBrainProcessor, BrainInput, BrainProcessor, BrainRuntime, ProcessingMode,
};

fn create_processor() -> ActualBrainProcessor {
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

fn create_input(text: &str, mode: ProcessingMode) -> BrainInput {
    BrainInput {
        text: text.into(),
        context: vec!["Previous context".into()],
        knowledge: vec![("brake_disc".into(), "braking component".into())],
        mode,
    }
}

#[test]
fn actual_brain_initializes() {
    let processor = create_processor();

    assert_eq!(
        processor
            .brain()
            .process(
                "What is a brake disc?",
                davinci::core::ProcessingMode::Inference
            )
            .unwrap()
            .rule_analysis
            .concept
            .as_deref(),
        Some("a brake disc")
    );
}

#[test]
fn actual_brain_processes_definition_question() {
    let processor = create_processor();

    let input = create_input("What is a brake disc?", ProcessingMode::Inference);

    let output = processor.process(&input).unwrap();

    assert!(output.response.contains("DefinitionQuestion"));
    assert!(output.response.contains("brake disc"));
}

#[test]
fn actual_brain_processes_statement() {
    let processor = create_processor();

    let input = create_input("Brake discs are important", ProcessingMode::Inference);

    let output = processor.process(&input).unwrap();

    assert!(output.response.contains("Statement"));
}

#[test]
fn actual_brain_exposes_neural_dimensions() {
    let processor = create_processor();

    let input = create_input("Engine temperature", ProcessingMode::Inference);

    let output = processor.process(&input).unwrap();

    assert!(output
        .response
        .contains("Neural representation dimensions: 8"));
}

#[test]
fn actual_brain_accepts_training_mode() {
    let processor = create_processor();

    let input = create_input("Brake thermal stress", ProcessingMode::Training);

    let output = processor.process(&input).unwrap();

    assert!(output.response.contains("Training"));
}

#[test]
fn actual_brain_rejects_empty_input() {
    let processor = create_processor();

    let input = create_input("", ProcessingMode::Inference);

    assert!(processor.process(&input).is_err());
}

#[test]
fn runtime_uses_actual_brain() {
    let path = format!("/tmp/davinci_phase8_{}.db", std::process::id());

    let _ = std::fs::remove_file(&path);

    let memory = MemoryController::new(8, &path).unwrap();

    let processor = create_processor();

    let mut runtime = BrainRuntime::new(memory, processor, ProcessingMode::Inference);

    let output = runtime.process("What is a brake disc?").unwrap();

    assert!(output.output.response.contains("DefinitionQuestion"));
    assert_eq!(output.mode, "INFERENCE");
}

#[test]
fn runtime_passes_retrieved_memory_to_processor() {
    let path = format!("/tmp/davinci_phase8_memory_{}.db", std::process::id());

    let _ = std::fs::remove_file(&path);

    let mut memory = MemoryController::new(8, &path).unwrap();

    memory
        .ltm
        .insert(
            davinci::memory::Knowledge::new("brake_disc".into(), "braking component".into())
                .unwrap(),
        )
        .unwrap();

    let processor = create_processor();

    let mut runtime = BrainRuntime::new(memory, processor, ProcessingMode::Inference);

    let output = runtime.process("brake_disc").unwrap();

    assert_eq!(output.retrieved_memories, 1);
    assert!(output
        .output
        .response
        .contains("Retrieved LTM knowledge items: 1"));
}
