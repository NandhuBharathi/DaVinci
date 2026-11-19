use super::monitor::StabilityState;
use super::sensors::SensorSnapshot;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrainingDecision {
    Normal,
    Warning,
    Critical,
}

pub struct TrainingBrainController;

impl TrainingBrainController {
    pub fn new() -> Self {
        Self
    }

    pub fn decide(
        &self,
        snapshot: &SensorSnapshot,
        monitor_state: StabilityState,
    ) -> TrainingDecision {
        if snapshot.stability == StabilityState::Critical
            || monitor_state == StabilityState::Critical
        {
            return TrainingDecision::Critical;
        }

        if snapshot.stability == StabilityState::Warning || monitor_state == StabilityState::Warning
        {
            return TrainingDecision::Warning;
        }

        TrainingDecision::Normal
    }
}

impl Default for TrainingBrainController {
    fn default() -> Self {
        Self::new()
    }
}

impl TrainingBrainController {
    pub fn adjust_learning_rate(
        &self,
        base_lr: f32,
        current_lr: f32,
        decision: TrainingDecision,
    ) -> Result<f32, String> {
        if !base_lr.is_finite() || !current_lr.is_finite() || base_lr <= 0.0 || current_lr <= 0.0 {
            return Err("Invalid learning rate".into());
        }

        let minimum_lr = base_lr * 0.1;

        match decision {
            TrainingDecision::Normal => Ok(current_lr),
            TrainingDecision::Warning => Ok((current_lr * 0.5).max(minimum_lr)),
            TrainingDecision::Critical => Ok(0.0),
        }
    }
}
