use crate::core::ProcessorKind;

use super::rules::Intent;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProcessingDecision {
    pub processors: Vec<ProcessorKind>,

    pub intent: Intent,
}

pub struct BrainController;

impl BrainController {
    pub fn new() -> Self {
        Self
    }

    pub fn decide(&self, intent: Intent) -> ProcessingDecision {
        let processors = vec![ProcessorKind::RuleBased, ProcessorKind::Neural];

        ProcessingDecision { processors, intent }
    }
}

impl Default for BrainController {
    fn default() -> Self {
        Self::new()
    }
}
