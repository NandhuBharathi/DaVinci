#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DataType {
    F32,
    F64,
    I32,
    I64,
    U8,
}

impl DataType {
    pub fn name(&self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F64 => "F64",
            Self::I32 => "I32",
            Self::I64 => "I64",
            Self::U8 => "U8",
        }
    }

    pub fn size_bytes(&self) -> usize {
        match self {
            Self::F32 | Self::I32 => 4,
            Self::F64 | Self::I64 => 8,
            Self::U8 => 1,
        }
    }

    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "F32" => Some(Self::F32),
            "F64" => Some(Self::F64),
            "I32" => Some(Self::I32),
            "I64" => Some(Self::I64),
            "U8" => Some(Self::U8),
            _ => None,
        }
    }
}
