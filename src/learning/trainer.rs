use crate::learning::dataset::LearningDataset;
use crate::learning::model::LearningModel;

pub struct TrainingConfig {
    pub epochs: usize,
    pub learning_rate: f32,
    pub hidden_size: usize,
}

pub struct TrainingReport {
    pub initial_loss: f32,
    pub final_loss: f32,
    pub steps: usize,
    pub weights_changed: bool,
}

pub struct LearningTrainer;

impl LearningTrainer {
    pub fn train(
        dataset: &LearningDataset,
        config: &TrainingConfig,
    ) -> Result<(LearningModel, TrainingReport), String> {
        if config.epochs == 0 || config.hidden_size == 0 {
            return Err("Epochs and hidden size must be non-zero".into());
        }

        if !config.learning_rate.is_finite() || config.learning_rate <= 0.0 {
            return Err("Invalid learning rate".into());
        }

        let (inputs, targets) = dataset.vocabulary();

        let mut model = LearningModel::new(inputs.len(), config.hidden_size, targets.len())?;

        let input_ids: std::collections::HashMap<&str, usize> = inputs
            .iter()
            .enumerate()
            .map(|(i, s)| (s.as_str(), i))
            .collect();

        let target_ids: std::collections::HashMap<&str, usize> = targets
            .iter()
            .enumerate()
            .map(|(i, s)| (s.as_str(), i))
            .collect();

        let initial_weights = model.embeddings.clone();

        let mut initial_loss = 0.0;
        let mut final_loss = 0.0;
        let mut steps = 0;

        for epoch in 0..config.epochs {
            let mut epoch_loss = 0.0;

            // Dataset order is preserved. No automatic shuffling.
            for pair in &dataset.pairs {
                let input_id = *input_ids
                    .get(pair.input.as_str())
                    .ok_or("Missing input vocabulary entry")?;

                let target_id = *target_ids
                    .get(pair.target.as_str())
                    .ok_or("Missing target vocabulary entry")?;

                let loss = model.train_step(input_id, target_id, config.learning_rate)?;

                epoch_loss += loss;
                steps += 1;
            }

            let average = epoch_loss / dataset.pairs.len() as f32;

            if epoch == 0 {
                initial_loss = average;
            }

            final_loss = average;
        }

        let weights_changed = model.embeddings != initial_weights;

        Ok((
            model,
            TrainingReport {
                initial_loss,
                final_loss,
                steps,
                weights_changed,
            },
        ))
    }
}
