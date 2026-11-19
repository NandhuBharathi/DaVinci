use super::data_types::DataType;
use super::device::DeviceConfig;

#[derive(Debug, Clone, PartialEq)]
pub enum TensorData {
    F32(Vec<f32>),
    F64(Vec<f64>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    U8(Vec<u8>),
}

impl TensorData {
    pub fn len(&self) -> usize {
        match self {
            Self::F32(v) => v.len(),
            Self::F64(v) => v.len(),
            Self::I32(v) => v.len(),
            Self::I64(v) => v.len(),
            Self::U8(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn data_type(&self) -> DataType {
        match self {
            Self::F32(_) => DataType::F32,
            Self::F64(_) => DataType::F64,
            Self::I32(_) => DataType::I32,
            Self::I64(_) => DataType::I64,
            Self::U8(_) => DataType::U8,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: TensorData,
    pub device: DeviceConfig,
}

impl Tensor {
    pub fn new(
        name: String,
        shape: Vec<usize>,
        data: TensorData,
        device: DeviceConfig,
    ) -> Result<Self, String> {
        let expected = shape
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .ok_or("Tensor shape overflow")?;

        if expected != data.len() {
            return Err(format!(
                "Shape mismatch: expected {}, received {}",
                expected,
                data.len()
            ));
        }

        Ok(Self {
            name,
            shape,
            data,
            device,
        })
    }

    pub fn num_elements(&self) -> usize {
        self.data.len()
    }

    pub fn data_type(&self) -> DataType {
        self.data.data_type()
    }

    pub fn memory_bytes(&self) -> usize {
        self.num_elements() * self.data_type().size_bytes()
    }
}
