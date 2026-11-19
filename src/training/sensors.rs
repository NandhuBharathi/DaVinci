use crate::learning::{
    GradientMonitor, GradientStatistics, LearningModel, WeightMonitor, WeightStatistics,
};

use crate::learning::model::TrainingStepDiagnostics;

use super::monitor::StabilityState;

#[derive(Debug, Clone)]
pub struct SensorConfig {
    pub warning_gradient_norm: f64,
    pub critical_gradient_norm: f64,

    pub warning_activation: f32,
    pub critical_activation: f32,

    pub max_memory_bytes: usize,
}

impl Default for SensorConfig {
    fn default() -> Self {
        Self {
            warning_gradient_norm: 5.0,
            critical_gradient_norm: 20.0,
            warning_activation: 10.0,
            critical_activation: 100.0,
            max_memory_bytes: 1024 * 1024 * 1024,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SensorSnapshot {
    pub step: usize,
    pub epoch: usize,

    pub loss: f32,

    pub weights: WeightStatistics,
    pub gradients: GradientStatistics,

    pub activation_count: usize,
    pub max_activation: f32,

    pub estimated_memory_bytes: usize,

    pub stability: StabilityState,
}

pub struct TrainingSensors {
    config: SensorConfig,
}

impl TrainingSensors {
    pub fn new(config: SensorConfig) -> Result<Self, String> {
        if !config.warning_gradient_norm.is_finite()
            || !config.critical_gradient_norm.is_finite()
            || config.warning_gradient_norm < 0.0
            || config.critical_gradient_norm <= config.warning_gradient_norm
            || !config.warning_activation.is_finite()
            || !config.critical_activation.is_finite()
            || config.warning_activation < 0.0
            || config.critical_activation <= config.warning_activation
            || config.max_memory_bytes == 0
        {
            return Err("Invalid sensor configuration".into());
        }

        Ok(Self { config })
    }

    pub fn inspect(
        &self,
        model: &LearningModel,
        diagnostics: &TrainingStepDiagnostics,
        step: usize,
        epoch: usize,
    ) -> Result<SensorSnapshot, String> {
        if !diagnostics.loss.is_finite() || diagnostics.loss < 0.0 {
            return Err("Loss sensor detected invalid loss".into());
        }

        if diagnostics.activations.is_empty()
            || diagnostics.activations.iter().any(|v| !v.is_finite())
        {
            return Err("Activation sensor detected invalid values".into());
        }

        let weights = WeightMonitor::validate(model)?;

        let gradients = GradientMonitor::inspect(&diagnostics.gradients)?;

        let max_activation = diagnostics
            .activations
            .iter()
            .map(|v| v.abs())
            .fold(0.0f32, f32::max);

        let weight_count = model
            .input_size
            .checked_mul(model.hidden_size)
            .and_then(|v| {
                model
                    .hidden_size
                    .checked_mul(model.output_size)
                    .and_then(|w| v.checked_add(w))
            })
            .and_then(|v| v.checked_add(model.output_size))
            .ok_or("Model memory size overflow")?;

        let estimated_memory_bytes = weight_count
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or("Model memory estimate overflow")?;

        let mut stability = StabilityState::Normal;

        if gradients.l2_norm >= self.config.critical_gradient_norm
            || max_activation >= self.config.critical_activation
            || estimated_memory_bytes > self.config.max_memory_bytes
        {
            stability = StabilityState::Critical;
        } else if gradients.l2_norm >= self.config.warning_gradient_norm
            || max_activation >= self.config.warning_activation
        {
            stability = StabilityState::Warning;
        }

        Ok(SensorSnapshot {
            step,
            epoch,
            loss: diagnostics.loss,
            weights,
            gradients,
            activation_count: diagnostics.activations.len(),
            max_activation,
            estimated_memory_bytes,
            stability,
        })
    }
}

impl Default for TrainingSensors {
    fn default() -> Self {
        Self::new(SensorConfig::default()).unwrap()
    }
}
