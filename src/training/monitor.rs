use crate::learning::{LearningModel, WeightMonitor};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StabilityState {
    Normal,
    Warning,
    Critical,
}

#[derive(Debug, Clone)]
pub struct TrainingMetrics {
    pub step: usize,
    pub epoch: usize,
    pub loss: f32,
    pub weight_norm: f64,
    pub max_weight: f32,
}

#[derive(Debug, Clone)]
pub struct MonitorConfig {
    pub warning_loss: f32,
    pub critical_loss: f32,
    pub max_weight_norm: f64,
    pub max_absolute_weight: f32,
}

impl Default for MonitorConfig {
    fn default() -> Self {
        Self {
            warning_loss: 5.0,
            critical_loss: 20.0,
            max_weight_norm: 1000.0,
            max_absolute_weight: 100.0,
        }
    }
}

pub struct TrainingMonitor {
    config: MonitorConfig,
}

impl TrainingMonitor {
    pub fn new(config: MonitorConfig) -> Result<Self, String> {
        if !config.warning_loss.is_finite()
            || !config.critical_loss.is_finite()
            || config.warning_loss < 0.0
            || config.critical_loss <= config.warning_loss
            || !config.max_weight_norm.is_finite()
            || config.max_weight_norm <= 0.0
            || !config.max_absolute_weight.is_finite()
            || config.max_absolute_weight <= 0.0
        {
            return Err("Invalid monitor configuration".into());
        }

        Ok(Self { config })
    }

    pub fn inspect(
        &self,
        model: &LearningModel,
        loss: f32,
        step: usize,
        epoch: usize,
    ) -> Result<(TrainingMetrics, StabilityState), String> {
        if !loss.is_finite() || loss < 0.0 {
            return Err("Invalid training loss".into());
        }

        let stats = WeightMonitor::validate(model)?;

        let max_weight = stats.min.abs().max(stats.max.abs());

        if stats.l2_norm > self.config.max_weight_norm
            || max_weight > self.config.max_absolute_weight
            || loss >= self.config.critical_loss
        {
            return Ok((
                TrainingMetrics {
                    step,
                    epoch,
                    loss,
                    weight_norm: stats.l2_norm,
                    max_weight,
                },
                StabilityState::Critical,
            ));
        }

        if loss >= self.config.warning_loss {
            return Ok((
                TrainingMetrics {
                    step,
                    epoch,
                    loss,
                    weight_norm: stats.l2_norm,
                    max_weight,
                },
                StabilityState::Warning,
            ));
        }

        Ok((
            TrainingMetrics {
                step,
                epoch,
                loss,
                weight_norm: stats.l2_norm,
                max_weight,
            },
            StabilityState::Normal,
        ))
    }
}

impl Default for TrainingMonitor {
    fn default() -> Self {
        Self::new(MonitorConfig::default()).unwrap()
    }
}
