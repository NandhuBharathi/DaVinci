use super::device::DeviceConfig;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcessingMode {
    Training,
    Inference,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcessorKind {
    RuleBased,
    Neural,
    SSM,
    Attention,
    KnowledgeGraph,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProcessingContext {
    pub mode: ProcessingMode,

    pub sequence_length: usize,

    pub device: DeviceConfig,

    pub processors: Vec<ProcessorKind>,
}

impl ProcessingContext {
    pub fn new(
        mode: ProcessingMode,
        sequence_length: usize,
        device: DeviceConfig,
    ) -> Result<Self, String> {
        if sequence_length == 0 {
            return Err("Sequence length cannot be zero".into());
        }

        Ok(Self {
            mode,
            sequence_length,
            device,
            processors: Vec::new(),
        })
    }

    pub fn add_processor(&mut self, processor: ProcessorKind) {
        if !self.processors.contains(&processor) {
            self.processors.push(processor);
        }
    }

    pub fn has_processor(&self, processor: ProcessorKind) -> bool {
        self.processors.contains(&processor)
    }
}
