use crate::brain::{IntegratedBrain, NeuralConfig};
use crate::core::{DeviceConfig, ProcessingMode as CoreProcessingMode};

use super::mode::ProcessingMode;
use super::processor::{BrainInput, BrainOutput, BrainProcessor};

pub struct ActualBrainProcessor {
    brain: IntegratedBrain,
}

impl ActualBrainProcessor {
    pub fn new(neural_config: NeuralConfig, device: DeviceConfig) -> Result<Self, String> {
        let brain = IntegratedBrain::new(neural_config, device)?;

        Ok(Self { brain })
    }

    pub fn train_and_install(
        &mut self,
        dataset: &crate::learning::LearningDataset,
        config: &crate::learning::TrainingConfig,
    ) -> Result<crate::learning::TrainingReport, String> {
        if dataset.pairs.is_empty() {
            return Err("Training dataset is empty".into());
        }

        let (input_vocabulary, target_vocabulary) = dataset.vocabulary();

        let (candidate_model, report) = crate::learning::LearningTrainer::train(dataset, config)?;

        if !report.final_loss.is_finite() {
            return Err("Training produced invalid loss".into());
        }

        if candidate_model
            .embeddings
            .iter()
            .flatten()
            .any(|v| !v.is_finite())
            || candidate_model
                .output_weights
                .iter()
                .flatten()
                .any(|v| !v.is_finite())
            || candidate_model.output_bias.iter().any(|v| !v.is_finite())
        {
            return Err("Training produced invalid weights".into());
        }

        // Install only after training and validation succeed.
        self.brain
            .install_learned_model(candidate_model, input_vocabulary, target_vocabulary)?;

        Ok(report)
    }

    pub fn brain(&self) -> &IntegratedBrain {
        &self.brain
    }
}

impl BrainProcessor for ActualBrainProcessor {
    fn process(&self, input: &BrainInput) -> Result<BrainOutput, String> {
        if input.text.trim().is_empty() {
            return Err("Brain input cannot be empty".into());
        }

        let core_mode = match input.mode {
            ProcessingMode::Inference => CoreProcessingMode::Inference,
            ProcessingMode::Training => CoreProcessingMode::Training,
        };

        // Execute the actual integrated Rule + Neural Brain.
        let result = self.brain.process(&input.text, core_mode)?;

        let intent = format!("{:?}", result.rule_analysis.intent);

        let concept = result.rule_analysis.concept.as_deref().unwrap_or("None");

        let neural_values = result.neural_representation.len();

        let learned_prediction = self.brain.predict_learned(&input.text)?;

        let learned_text = learned_prediction
            .as_deref()
            .unwrap_or("No learned prediction");

        let response = format!(
            "Intent: {}\n\
             Concept: {}\n\
             Tokens: {}\n\
             Neural representation dimensions: {}\n\
             Processors: {:?}\n\
             Retrieved STM context items: {}\n\
             Retrieved LTM knowledge items: {}\n\
             Processing mode: {:?}\\n\\
             Learned prediction: {}",
            intent,
            concept,
            result.rule_analysis.tokens.len(),
            neural_values,
            result.decision.processors,
            input.context.len(),
            input.knowledge.len(),
            input.mode,
            learned_text,
        );

        // This is a neutral placeholder, NOT calibrated model confidence.
        let confidence = 0.5;

        Ok(BrainOutput {
            response,
            confidence,
        })
    }
}
