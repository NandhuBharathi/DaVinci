#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BrainState {
    Idle,

    Training { epoch: u64, step: u64 },

    Inference,

    Paused,

    Error { message: String },
}

impl BrainState {
    pub fn is_active(&self) -> bool {
        matches!(self, Self::Training { .. } | Self::Inference)
    }

    pub fn is_safe(&self) -> bool {
        !matches!(self, Self::Error { .. })
    }
}
