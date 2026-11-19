use super::processor::BrainOutput;

pub struct RuntimeResult {
    pub output: BrainOutput,
    pub retrieved_memories: usize,
    pub mode: String,
}
