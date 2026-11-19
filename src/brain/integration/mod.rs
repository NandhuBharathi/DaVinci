use crate::core::{DeviceConfig, ProcessingContext, ProcessingMode};

use super::{
    BrainController, NeuralConfig, NeuralEngine, ProcessingDecision, RuleAnalysis, RuleEngine,
};

pub struct BrainOutput {
    pub rule_analysis: RuleAnalysis,

    pub neural_representation: Vec<f32>,

    pub decision: ProcessingDecision,

    pub context: ProcessingContext,
}

pub struct IntegratedBrain {
    rule_engine: RuleEngine,

    neural_engine: NeuralEngine,

    controller: BrainController,

    device: DeviceConfig,
}

impl IntegratedBrain {
    pub fn new(neural_config: NeuralConfig, device: DeviceConfig) -> Result<Self, String> {
        Ok(Self {
            rule_engine: RuleEngine::new(),

            neural_engine: NeuralEngine::new(neural_config)?,

            controller: BrainController::new(),

            device,
        })
    }

    pub fn install_learned_model(
        &mut self,
        model: crate::learning::LearningModel,
        input_vocabulary: Vec<String>,
        target_vocabulary: Vec<String>,
    ) -> Result<(), String> {
        self.neural_engine
            .install_learned_model(model, input_vocabulary, target_vocabulary)
    }

    pub fn predict_learned(&self, input: &str) -> Result<Option<String>, String> {
        self.neural_engine.predict_learned(input)
    }

    pub fn has_learned_model(&self) -> bool {
        self.neural_engine.has_learned_model()
    }

    pub fn process(&self, input: &str, mode: ProcessingMode) -> Result<BrainOutput, String> {
        // Rule Processing

        let rule_analysis = self.rule_engine.analyze(input);

        // Neural Input Representation

        let features = self.neural_engine.encode_text(input);

        // Neural Computation

        let neural_representation = self.neural_engine.forward(&features)?;

        // Brain Decision

        let decision = self.controller.decide(rule_analysis.intent.clone());

        // Processing Context

        let mut context =
            ProcessingContext::new(mode, rule_analysis.tokens.len().max(1), self.device.clone())?;

        for processor in &decision.processors {
            context.add_processor(*processor);
        }

        Ok(BrainOutput {
            rule_analysis,

            neural_representation,

            decision,

            context,
        })
    }
}
