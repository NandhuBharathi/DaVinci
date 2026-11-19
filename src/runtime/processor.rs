use super::mode::ProcessingMode;

pub struct BrainInput {
    pub text: String,
    pub context: Vec<String>,
    pub knowledge: Vec<(String, String)>,
    pub mode: ProcessingMode,
}

pub struct BrainOutput {
    pub response: String,
    pub confidence: f32,
}

pub trait BrainProcessor {
    fn process(&self, input: &BrainInput) -> Result<BrainOutput, String>;
}
