use crate::learning::LearningModel;

#[derive(Debug, Clone, PartialEq)]
pub struct WeightStatistics {
    pub count: usize,
    pub min: f32,
    pub max: f32,
    pub mean: f32,
    pub l2_norm: f64,
}

pub struct WeightRegistry;

impl WeightRegistry {
    pub fn collect(model: &LearningModel) -> Result<WeightStatistics, String> {
        let values: Vec<f32> = model
            .embeddings
            .iter()
            .flatten()
            .chain(model.output_weights.iter().flatten())
            .chain(model.output_bias.iter())
            .copied()
            .collect();

        if values.is_empty() {
            return Err("No weights available".into());
        }

        if values.iter().any(|v| !v.is_finite()) {
            return Err("Non-finite weight detected".into());
        }

        let min = values.iter().copied().fold(f32::INFINITY, f32::min);
        let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        let sum: f64 = values.iter().map(|v| *v as f64).sum();

        let squared_sum: f64 = values.iter().map(|v| (*v as f64) * (*v as f64)).sum();

        Ok(WeightStatistics {
            count: values.len(),
            min,
            max,
            mean: (sum / values.len() as f64) as f32,
            l2_norm: squared_sum.sqrt(),
        })
    }
}

pub struct WeightMonitor;

impl WeightMonitor {
    pub fn validate(model: &LearningModel) -> Result<WeightStatistics, String> {
        if model.input_size == 0 || model.hidden_size == 0 || model.output_size == 0 {
            return Err("Invalid model dimensions".into());
        }

        if model.embeddings.len() != model.input_size
            || model.output_weights.len() != model.hidden_size
            || model.output_bias.len() != model.output_size
        {
            return Err("Weight dimensions mismatch".into());
        }

        for row in &model.embeddings {
            if row.len() != model.hidden_size {
                return Err("Embedding shape mismatch".into());
            }
        }

        for row in &model.output_weights {
            if row.len() != model.output_size {
                return Err("Output weight shape mismatch".into());
            }
        }

        WeightRegistry::collect(model)
    }
}

pub struct GradientMonitor;

#[derive(Debug, Clone, PartialEq)]
pub struct GradientStatistics {
    pub count: usize,
    pub max_absolute: f32,
    pub l2_norm: f64,
}

impl GradientMonitor {
    pub fn inspect(gradients: &[f32]) -> Result<GradientStatistics, String> {
        if gradients.is_empty() {
            return Err("Gradient array is empty".into());
        }

        if gradients.iter().any(|g| !g.is_finite()) {
            return Err("Non-finite gradient detected".into());
        }

        let max_absolute = gradients.iter().map(|g| g.abs()).fold(0.0f32, f32::max);

        let l2_norm = gradients
            .iter()
            .map(|g| (*g as f64).powi(2))
            .sum::<f64>()
            .sqrt();

        Ok(GradientStatistics {
            count: gradients.len(),
            max_absolute,
            l2_norm,
        })
    }

    pub fn clip(gradients: &mut [f32], max_norm: f32) -> Result<(), String> {
        if !max_norm.is_finite() || max_norm <= 0.0 {
            return Err("Invalid gradient clipping threshold".into());
        }

        let stats = Self::inspect(gradients)?;

        if stats.l2_norm > max_norm as f64 {
            let scale = max_norm as f64 / stats.l2_norm;

            for gradient in gradients {
                *gradient = (*gradient as f64 * scale) as f32;
            }
        }

        Ok(())
    }
}
