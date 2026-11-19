use crate::memory::MemoryController;

use super::mode::ProcessingMode;
use super::processor::{BrainInput, BrainOutput, BrainProcessor};
use super::result::RuntimeResult;

pub struct BrainRuntime<P: BrainProcessor> {
    pub memory: MemoryController,
    pub processor: P,
    pub mode: ProcessingMode,
}

impl<P: BrainProcessor> BrainRuntime<P> {
    pub fn new(memory: MemoryController, processor: P, mode: ProcessingMode) -> Self {
        Self {
            memory,
            processor,
            mode,
        }
    }

    pub fn set_mode(&mut self, mode: ProcessingMode) {
        self.mode = mode;
    }

    pub fn process(&mut self, text: &str) -> Result<RuntimeResult, String> {
        if text.trim().is_empty() {
            return Err("Input cannot be empty".into());
        }

        // 1. Store current input in STM.
        self.memory.stm.remember("current_input", text)?;

        // 2. Retrieve relevant LTM knowledge.
        let memories = self.memory.ltm.search(text);

        let knowledge: Vec<(String, String)> = memories
            .iter()
            .map(|item| (item.key.clone(), item.value.clone()))
            .collect();

        // 3. Collect recent STM context.
        let context: Vec<String> = self
            .memory
            .stm
            .store
            .entries()
            .iter()
            .map(|entry| format!("{}: {}", entry.key, entry.value))
            .collect();

        // 4. Prepare unified Brain input.
        let brain_input = BrainInput {
            text: text.to_string(),
            context,
            knowledge,
            mode: self.mode,
        };

        // 5. Process through Brain.
        let output: BrainOutput = self.processor.process(&brain_input)?;

        if !output.confidence.is_finite() || !(0.0..=1.0).contains(&output.confidence) {
            return Err("Invalid Brain confidence".into());
        }

        if output.response.trim().is_empty() {
            return Err("Brain returned an empty response".into());
        }

        // 6. Store response in STM.
        self.memory
            .stm
            .remember("last_response", &output.response)?;

        Ok(RuntimeResult {
            output,
            retrieved_memories: memories.len(),
            mode: match self.mode {
                ProcessingMode::Inference => "INFERENCE".into(),
                ProcessingMode::Training => "TRAINING".into(),
            },
        })
    }

    // Explicit consolidation only. Never automatic.
    pub fn consolidate_stm(&mut self, key: &str) -> Result<(), String> {
        self.memory.consolidate(key)
    }
}
