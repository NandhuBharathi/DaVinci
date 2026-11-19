use crate::learning::GradientController;
use crate::learning::OptimizerConfig;
use crate::learning::OptimizerEngine;
use crate::learning::OptimizerKind;

#[derive(Debug, Clone)]
pub struct LearningModel {
    pub input_size: usize,
    pub hidden_size: usize,
    pub output_size: usize,
    pub embeddings: Vec<Vec<f32>>,
    pub output_weights: Vec<Vec<f32>>,
    pub output_bias: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct TrainingStepDiagnostics {
    pub loss: f32,
    pub gradients: Vec<f32>,
    pub activations: Vec<f32>,
}

impl LearningModel {
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Result<Self, String> {
        if input_size == 0 || hidden_size == 0 || output_size == 0 {
            return Err("Model dimensions must be non-zero".into());
        }

        let mut embeddings = vec![vec![0.0; hidden_size]; input_size];
        let mut output_weights = vec![vec![0.0; output_size]; hidden_size];

        for i in 0..input_size {
            for j in 0..hidden_size {
                embeddings[i][j] = (((i * 17 + j * 13 + 7) % 101) as f32 - 50.0) * 0.001;
            }
        }

        for i in 0..hidden_size {
            for j in 0..output_size {
                output_weights[i][j] = (((i * 19 + j * 11 + 3) % 97) as f32 - 48.0) * 0.001;
            }
        }

        Ok(Self {
            input_size,
            hidden_size,
            output_size,
            embeddings,
            output_weights,
            output_bias: vec![0.0; output_size],
        })
    }

    pub fn probabilities(&self, input_id: usize) -> Result<Vec<f32>, String> {
        if input_id >= self.input_size {
            return Err("Input ID out of range".into());
        }

        let embedding = &self.embeddings[input_id];
        let mut logits = vec![0.0; self.output_size];

        for (h, value) in embedding.iter().enumerate() {
            for (o, logit) in logits.iter_mut().enumerate() {
                *logit += value * self.output_weights[h][o];
            }
        }

        for (logit, bias) in logits.iter_mut().zip(&self.output_bias) {
            *logit += bias;
        }

        let max_value = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut probabilities: Vec<f32> = logits.iter().map(|x| (x - max_value).exp()).collect();

        let total: f32 = probabilities.iter().sum();

        if !total.is_finite() || total <= 0.0 {
            return Err("Invalid probability distribution".into());
        }

        for probability in &mut probabilities {
            *probability /= total;
        }

        Ok(probabilities)
    }

    // Calculates gradients without changing model parameters.
    pub fn calculate_gradients(
        &self,
        input_id: usize,
        target_id: usize,
        controller: &GradientController,
    ) -> Result<
        (
            TrainingStepDiagnostics,
            crate::learning::GradientControlReport,
        ),
        String,
    > {
        if input_id >= self.input_size || target_id >= self.output_size {
            return Err("Training ID out of range".into());
        }

        let probabilities = self.probabilities(input_id)?;
        let loss = -probabilities[target_id].max(1e-9).ln();

        if !loss.is_finite() {
            return Err("Non-finite training loss".into());
        }

        let activations = self.embeddings[input_id].clone();

        if activations.iter().any(|v| !v.is_finite()) {
            return Err("Non-finite activation".into());
        }

        let mut output_gradient = probabilities;
        output_gradient[target_id] -= 1.0;

        let mut embedding_gradient = vec![0.0; self.hidden_size];

        for h in 0..self.hidden_size {
            for o in 0..self.output_size {
                embedding_gradient[h] += self.output_weights[h][o] * output_gradient[o];
            }
        }

        // Gradient layout:
        // 1. Active input embedding row
        // 2. All output weights
        // 3. Output bias
        // No duplicate parameter gradients.
        let mut gradients = Vec::with_capacity(
            self.hidden_size + self.hidden_size * self.output_size + self.output_size,
        );

        gradients.extend(embedding_gradient);

        for h in 0..self.hidden_size {
            for o in 0..self.output_size {
                gradients.push(activations[h] * output_gradient[o]);
            }
        }

        gradients.extend(output_gradient);

        let (clipped, report) = controller.process(&gradients)?;

        Ok((
            TrainingStepDiagnostics {
                loss,
                gradients: clipped,
                activations,
            },
            report,
        ))
    }

    // Compatibility method: uses the original SGD-style update.
    pub fn train_step_with_gradient_control(
        &mut self,
        input_id: usize,
        target_id: usize,
        learning_rate: f32,
        controller: &GradientController,
    ) -> Result<
        (
            TrainingStepDiagnostics,
            crate::learning::GradientControlReport,
        ),
        String,
    > {
        if !learning_rate.is_finite() || learning_rate <= 0.0 {
            return Err("Invalid learning rate".into());
        }

        let (diagnostics, report) = self.calculate_gradients(input_id, target_id, controller)?;

        if report.exploded {
            return Ok((diagnostics, report));
        }

        let mut optimizer =
            OptimizerEngine::new(OptimizerConfig::new(OptimizerKind::Sgd, learning_rate)?)?;

        let config = controller.config();

        optimizer.update(
            self,
            input_id,
            &diagnostics.gradients,
            config.freeze_embeddings,
            config.freeze_output_weights,
            config.freeze_output_bias,
        )?;

        Ok((diagnostics, report))
    }

    pub fn train_step_with_diagnostics(
        &mut self,
        input_id: usize,
        target_id: usize,
        learning_rate: f32,
    ) -> Result<TrainingStepDiagnostics, String> {
        let controller = GradientController::default();

        let (diagnostics, report) =
            self.train_step_with_gradient_control(input_id, target_id, learning_rate, &controller)?;

        if report.exploded {
            return Err("Gradient explosion detected; update rejected".into());
        }

        Ok(diagnostics)
    }

    pub fn train_step(
        &mut self,
        input_id: usize,
        target_id: usize,
        learning_rate: f32,
    ) -> Result<f32, String> {
        Ok(self
            .train_step_with_diagnostics(input_id, target_id, learning_rate)?
            .loss)
    }
}
