use super::device::DeviceConfig;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    FP32,
    FP64,
    FP16,
    BF16,
    INT8,
    INT4,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ModelConfig {
    pub model_name: String,

    pub vocab_size: usize,

    pub hidden_size: usize,

    pub num_layers: usize,

    pub max_sequence_length: usize,

    pub precision: Precision,

    pub device: DeviceConfig,
}

impl ModelConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.model_name.trim().is_empty() {
            return Err("Model name cannot be empty".into());
        }

        if self.vocab_size == 0 {
            return Err("Vocabulary size must be greater than zero".into());
        }

        if self.hidden_size == 0 {
            return Err("Hidden size must be greater than zero".into());
        }

        if self.num_layers == 0 {
            return Err("Number of layers must be greater than zero".into());
        }

        if self.max_sequence_length == 0 {
            return Err("Maximum sequence length must be greater than zero".into());
        }

        Ok(())
    }
}
