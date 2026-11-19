use crate::learning::LearningModel;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NeuralConfig {
    pub input_size: usize,
    pub hidden_size: usize,
    pub output_size: usize,
}

impl NeuralConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.input_size == 0 || self.hidden_size == 0 || self.output_size == 0 {
            return Err("Neural dimensions must be greater than zero".into());
        }

        let w1 = self.input_size.checked_mul(self.hidden_size);

        let w2 = self.hidden_size.checked_mul(self.output_size);

        if w1.is_none() || w2.is_none() {
            return Err("Neural dimensions overflow".into());
        }

        Ok(())
    }
}

struct LearnedHead {
    model: LearningModel,
    input_vocabulary: Vec<String>,
    target_vocabulary: Vec<String>,
}

pub struct NeuralEngine {
    config: NeuralConfig,

    weights_1: Vec<f32>,

    bias_1: Vec<f32>,

    weights_2: Vec<f32>,

    bias_2: Vec<f32>,

    learned_head: Option<LearnedHead>,
}

impl NeuralEngine {
    pub fn new(config: NeuralConfig) -> Result<Self, String> {
        config.validate()?;

        let w1_size = config
            .input_size
            .checked_mul(config.hidden_size)
            .ok_or("First layer size overflow")?;

        let w2_size = config
            .hidden_size
            .checked_mul(config.output_size)
            .ok_or("Second layer size overflow")?;

        let weights_1 = Self::initialize_weights(w1_size);

        let weights_2 = Self::initialize_weights(w2_size);

        Ok(Self {
            bias_1: vec![0.0; config.hidden_size],

            bias_2: vec![0.0; config.output_size],

            weights_1,

            weights_2,

            config,
            learned_head: None,
        })
    }

    fn initialize_weights(size: usize) -> Vec<f32> {
        (0..size)
            .map(|i| {
                let value = ((i * 37 + 17) % 2001) as f32;

                (value / 1000.0 - 1.0) * 0.05
            })
            .collect()
    }

    pub fn forward(&self, input: &[f32]) -> Result<Vec<f32>, String> {
        if input.len() != self.config.input_size {
            return Err(format!(
                "Input dimension mismatch: expected {}, received {}",
                self.config.input_size,
                input.len()
            ));
        }

        // First Dense Layer

        let mut hidden = vec![0.0; self.config.hidden_size];

        for j in 0..self.config.hidden_size {
            let mut sum = self.bias_1[j];

            for i in 0..self.config.input_size {
                sum += input[i] * self.weights_1[j * self.config.input_size + i];
            }

            hidden[j] = sum.tanh();
        }

        // Second Dense Layer

        let mut output = vec![0.0; self.config.output_size];

        for k in 0..self.config.output_size {
            let mut sum = self.bias_2[k];

            for j in 0..self.config.hidden_size {
                sum += hidden[j] * self.weights_2[k * self.config.hidden_size + j];
            }

            output[k] = sum.tanh();
        }

        Ok(output)
    }

    pub fn encode_text(&self, text: &str) -> Vec<f32> {
        let mut features = vec![0.0; self.config.input_size];

        if features.is_empty() {
            return features;
        }

        for token in text.split_whitespace() {
            let mut hash: u64 = 2166136261;

            for byte in token.as_bytes() {
                hash ^= *byte as u64;

                hash = hash.wrapping_mul(16777619);
            }

            let index = (hash as usize) % self.config.input_size;

            features[index] += 1.0;
        }

        let count = text.split_whitespace().count();

        if count > 0 {
            for value in &mut features {
                *value /= count as f32;
            }
        }

        features
    }

    pub fn install_learned_model(
        &mut self,
        model: LearningModel,
        input_vocabulary: Vec<String>,
        target_vocabulary: Vec<String>,
    ) -> Result<(), String> {
        if model.input_size != input_vocabulary.len()
            || model.output_size != target_vocabulary.len()
        {
            return Err("Learned model vocabulary dimensions mismatch".into());
        }

        if model.hidden_size == 0
            || model.embeddings.len() != model.input_size
            || model.output_weights.len() != model.hidden_size
            || model.output_bias.len() != model.output_size
        {
            return Err("Invalid learned model dimensions".into());
        }

        if model
            .embeddings
            .iter()
            .any(|row| row.len() != model.hidden_size || row.iter().any(|v| !v.is_finite()))
        {
            return Err("Invalid embedding weights".into());
        }

        if model
            .output_weights
            .iter()
            .any(|row| row.len() != model.output_size || row.iter().any(|v| !v.is_finite()))
        {
            return Err("Invalid output weights".into());
        }

        if model.output_bias.iter().any(|v| !v.is_finite()) {
            return Err("Invalid output bias".into());
        }

        if input_vocabulary.iter().any(|v| v.trim().is_empty())
            || target_vocabulary.iter().any(|v| v.trim().is_empty())
        {
            return Err("Vocabulary cannot contain empty entries".into());
        }

        if input_vocabulary
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len()
            != input_vocabulary.len()
            || target_vocabulary
                .iter()
                .collect::<std::collections::HashSet<_>>()
                .len()
                != target_vocabulary.len()
        {
            return Err("Duplicate vocabulary entries".into());
        }

        self.learned_head = Some(LearnedHead {
            model,
            input_vocabulary,
            target_vocabulary,
        });

        Ok(())
    }

    pub fn predict_learned(&self, text: &str) -> Result<Option<String>, String> {
        let Some(head) = &self.learned_head else {
            return Ok(None);
        };

        let mut accumulated = vec![0.0f32; head.target_vocabulary.len()];
        let mut matched = 0usize;

        for token in text.split_whitespace() {
            let normalized = token
                .trim_matches(|c: char| !c.is_alphanumeric())
                .to_lowercase();

            if normalized.is_empty() {
                continue;
            }

            let Some(input_id) = head
                .input_vocabulary
                .iter()
                .position(|word| word.to_lowercase() == normalized)
            else {
                continue;
            };

            let probabilities = head.model.probabilities(input_id)?;

            for (total, probability) in accumulated.iter_mut().zip(probabilities) {
                *total += probability;
            }

            matched += 1;
        }

        if matched == 0 {
            return Ok(None);
        }

        for value in &mut accumulated {
            *value /= matched as f32;
        }

        let best_id = accumulated
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(id, _)| id)
            .ok_or("No learned prediction available")?;

        Ok(Some(head.target_vocabulary[best_id].clone()))
    }

    pub fn has_learned_model(&self) -> bool {
        self.learned_head.is_some()
    }

    pub fn config(&self) -> &NeuralConfig {
        &self.config
    }
}
