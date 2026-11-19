#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeviceConfig {
    CPU,
    CUDA { device_id: usize },
}

impl DeviceConfig {
    pub fn name(&self) -> &'static str {
        match self {
            Self::CPU => "CPU",
            Self::CUDA { .. } => "CUDA",
        }
    }

    pub fn device_id(&self) -> Option<usize> {
        match self {
            Self::CPU => None,
            Self::CUDA { device_id } => Some(*device_id),
        }
    }
}
