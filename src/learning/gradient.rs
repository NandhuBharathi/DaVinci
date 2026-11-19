use super::weights::{GradientMonitor, GradientStatistics};

#[derive(Debug, Clone)]
pub struct GradientControlConfig {
    pub max_norm: f32,
    pub warning_norm: f32,
    pub critical_norm: f32,
    pub freeze_embeddings: bool,
    pub freeze_output_weights: bool,
    pub freeze_output_bias: bool,
}

impl Default for GradientControlConfig {
    fn default() -> Self {
        Self {
            max_norm: 5.0,
            warning_norm: 5.0,
            critical_norm: 20.0,
            freeze_embeddings: false,
            freeze_output_weights: false,
            freeze_output_bias: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GradientControlReport {
    pub raw: GradientStatistics,
    pub clipped: GradientStatistics,
    pub clipping_scale: f32,
    pub exploded: bool,
}

pub struct GradientController {
    config: GradientControlConfig,
}

impl GradientController {
    pub fn new(config: GradientControlConfig) -> Result<Self, String> {
        if !config.max_norm.is_finite()
            || !config.warning_norm.is_finite()
            || !config.critical_norm.is_finite()
            || config.max_norm <= 0.0
            || config.warning_norm < 0.0
            || config.critical_norm <= config.warning_norm
        {
            return Err("Invalid gradient controller configuration".into());
        }

        Ok(Self { config })
    }

    pub fn config(&self) -> &GradientControlConfig {
        &self.config
    }

    pub fn process(&self, gradients: &[f32]) -> Result<(Vec<f32>, GradientControlReport), String> {
        let raw = GradientMonitor::inspect(gradients)?;

        let exploded = raw.l2_norm >= self.config.critical_norm as f64;

        let mut clipped_values = gradients.to_vec();

        let mut clipping_scale = 1.0f32;

        if raw.l2_norm > self.config.max_norm as f64 {
            clipping_scale = (self.config.max_norm as f64 / raw.l2_norm) as f32;

            for value in &mut clipped_values {
                *value *= clipping_scale;
            }
        }

        let clipped = GradientMonitor::inspect(&clipped_values)?;

        Ok((
            clipped_values,
            GradientControlReport {
                raw,
                clipped,
                clipping_scale,
                exploded,
            },
        ))
    }
}

impl Default for GradientController {
    fn default() -> Self {
        Self::new(GradientControlConfig::default()).unwrap()
    }
}
