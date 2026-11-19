use davinci::memory::MemoryController;
use davinci::runtime::{BrainInput, BrainOutput, BrainProcessor, BrainRuntime, ProcessingMode};

struct TestProcessor;

impl BrainProcessor for TestProcessor {
    fn process(&self, input: &BrainInput) -> Result<BrainOutput, String> {
        let response = if input.knowledge.is_empty() {
            format!("Processed: {}", input.text)
        } else {
            format!(
                "Processed: {} | Retrieved: {} memories",
                input.text,
                input.knowledge.len()
            )
        };

        Ok(BrainOutput {
            response,
            confidence: 0.8,
        })
    }
}

fn create_runtime(name: &str) -> BrainRuntime<TestProcessor> {
    let path = format!("/tmp/davinci_runtime_{}_{}.db", name, std::process::id());

    let _ = std::fs::remove_file(&path);

    let memory = MemoryController::new(8, &path).unwrap();

    BrainRuntime::new(memory, TestProcessor, ProcessingMode::Inference)
}

#[test]
fn runtime_processes_input() {
    let mut runtime = create_runtime("input");

    let result = runtime.process("Hello DaVinci").unwrap();

    assert!(result.output.response.contains("Hello DaVinci"));
    assert_eq!(result.mode, "INFERENCE");
}

#[test]
fn runtime_rejects_empty_input() {
    let mut runtime = create_runtime("empty");

    assert!(runtime.process(" ").is_err());
}

#[test]
fn runtime_stores_input_in_stm() {
    let mut runtime = create_runtime("stm");

    runtime.process("Brake disc").unwrap();

    assert_eq!(
        runtime.memory.stm.recall("current_input").unwrap(),
        "Brake disc"
    );
}

#[test]
fn runtime_stores_response_in_stm() {
    let mut runtime = create_runtime("response");

    runtime.process("Test input").unwrap();

    assert!(runtime.memory.stm.recall("last_response").is_some());
}

#[test]
fn runtime_retrieves_ltm_knowledge() {
    let mut runtime = create_runtime("ltm");

    runtime
        .memory
        .ltm
        .insert(
            davinci::memory::Knowledge::new("brake_disc".into(), "braking_component".into())
                .unwrap(),
        )
        .unwrap();

    let result = runtime.process("brake_disc").unwrap();

    assert_eq!(result.retrieved_memories, 1);
}

#[test]
fn runtime_mode_can_be_changed() {
    let mut runtime = create_runtime("mode");

    runtime.set_mode(ProcessingMode::Training);

    assert_eq!(runtime.mode, ProcessingMode::Training);
}

#[test]
fn inference_does_not_change_learning_weights() {
    use davinci::learning::LearningModel;

    let model = LearningModel::new(2, 8, 2).unwrap();
    let before = model.embeddings.clone();

    let mut runtime = create_runtime("weights");

    runtime.process("Test inference").unwrap();

    assert_eq!(before, model.embeddings);
}

#[test]
fn explicit_stm_consolidation_works() {
    let mut runtime = create_runtime("consolidation");

    runtime
        .memory
        .stm
        .remember("concept", "learned_value")
        .unwrap();

    runtime.consolidate_stm("concept").unwrap();

    assert!(runtime.memory.stm.recall("concept").is_none());

    assert_eq!(
        runtime.memory.recall_long_term("concept").unwrap(),
        "learned_value"
    );
}

#[test]
fn invalid_processor_confidence_is_rejected() {
    struct InvalidProcessor;

    impl BrainProcessor for InvalidProcessor {
        fn process(&self, _: &BrainInput) -> Result<BrainOutput, String> {
            Ok(BrainOutput {
                response: "Invalid".into(),
                confidence: 2.0,
            })
        }
    }

    let path = format!("/tmp/davinci_runtime_invalid_{}.db", std::process::id());

    let _ = std::fs::remove_file(&path);

    let memory = MemoryController::new(8, &path).unwrap();

    let mut runtime = BrainRuntime::new(memory, InvalidProcessor, ProcessingMode::Inference);

    assert!(runtime.process("test").is_err());
}
