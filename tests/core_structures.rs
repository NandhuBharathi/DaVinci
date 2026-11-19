use davinci::core::*;

#[test]
fn tensor_creation() {
    let tensor = Tensor::new(
        "input".into(),
        vec![2, 2],
        TensorData::F32(vec![1.0, 2.0, 3.0, 4.0]),
        DeviceConfig::CPU,
    );

    assert!(tensor.is_ok());
}

#[test]
fn tensor_shape_validation() {
    let tensor = Tensor::new(
        "invalid".into(),
        vec![2, 3],
        TensorData::F32(vec![1.0, 2.0]),
        DeviceConfig::CPU,
    );

    assert!(tensor.is_err());
}

#[test]
fn tensor_memory_calculation() {
    let tensor = Tensor::new(
        "memory".into(),
        vec![4],
        TensorData::F32(vec![1.0, 2.0, 3.0, 4.0]),
        DeviceConfig::CPU,
    )
    .unwrap();

    assert_eq!(tensor.memory_bytes(), 16);
}

#[test]
fn datatype_registry() {
    assert_eq!(DataType::parse("F32"), Some(DataType::F32));
    assert_eq!(DataType::parse("F64"), Some(DataType::F64));
    assert_eq!(DataType::parse("UNKNOWN"), None);
}

#[test]
fn device_configuration() {
    let device = DeviceConfig::CUDA { device_id: 0 };

    assert_eq!(device.name(), "CUDA");
    assert_eq!(device.device_id(), Some(0));
}

#[test]
fn model_configuration_validation() {
    let config = ModelConfig {
        model_name: "DaVinci".into(),
        vocab_size: 256,
        hidden_size: 128,
        num_layers: 4,
        max_sequence_length: 512,
        precision: Precision::FP32,
        device: DeviceConfig::CPU,
    };

    assert!(config.validate().is_ok());
}

#[test]
fn brain_state_monitoring() {
    let state = BrainState::Training {
        epoch: 1,
        step: 100,
    };

    assert!(state.is_active());
    assert!(state.is_safe());
}

#[test]
fn processing_context() {
    let mut context =
        ProcessingContext::new(ProcessingMode::Training, 128, DeviceConfig::CPU).unwrap();

    context.add_processor(ProcessorKind::RuleBased);
    context.add_processor(ProcessorKind::SSM);

    assert!(context.has_processor(ProcessorKind::SSM));
    assert!(!context.has_processor(ProcessorKind::Attention));
}

#[test]
fn invalid_processing_context() {
    let context = ProcessingContext::new(ProcessingMode::Training, 0, DeviceConfig::CPU);

    assert!(context.is_err());
}
