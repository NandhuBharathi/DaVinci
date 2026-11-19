use crate::learning::{
    CheckpointManager, GradientController, LearningDataset, LearningModel, LearningRateScheduler,
    OptimizerConfig, OptimizerEngine, OptimizerKind, SchedulerConfig, SchedulerKind,
    TrainingConfig,
};

use super::brain_controller::{TrainingBrainController, TrainingDecision};
use super::monitor::TrainingMonitor;
use super::sensors::TrainingSensors;

use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrainingStatus {
    Completed,
    PausedAtStepLimit,
    StoppedCritical,
}

#[derive(Debug, Clone)]
pub struct TrainingReport {
    pub status: TrainingStatus,
    pub steps: usize,
    pub completed_epochs: usize,
    pub initial_loss: f32,
    pub final_loss: f32,
    pub best_loss: f32,
    pub warnings: usize,
    pub critical_events: usize,
}

pub struct ControllerConfig {
    pub max_steps: Option<usize>,
    pub monitor: TrainingMonitor,
}

pub struct TrainingController {
    pub checkpoints: CheckpointManager,
    pub config: ControllerConfig,
}

impl TrainingController {
    pub fn new(checkpoints: CheckpointManager, config: ControllerConfig) -> Self {
        Self {
            checkpoints,
            config,
        }
    }

    pub fn train(
        &self,
        dataset: &LearningDataset,
        training: &TrainingConfig,
    ) -> Result<(LearningModel, TrainingReport), String> {
        self.train_with_optimizer_and_scheduler(dataset, training, "sgd", "constant")
    }

    pub fn train_with_optimizer(
        &self,
        dataset: &LearningDataset,
        training: &TrainingConfig,
        optimizer_keyword: &str,
    ) -> Result<(LearningModel, TrainingReport), String> {
        self.train_with_optimizer_and_scheduler(dataset, training, optimizer_keyword, "constant")
    }

    pub fn train_with_optimizer_and_scheduler(
        &self,
        dataset: &LearningDataset,
        training: &TrainingConfig,
        optimizer_keyword: &str,
        scheduler_keyword: &str,
    ) -> Result<(LearningModel, TrainingReport), String> {
        if dataset.pairs.is_empty() {
            return Err("Training dataset is empty".into());
        }

        if training.epochs == 0 || training.hidden_size == 0 {
            return Err("Invalid training dimensions".into());
        }

        if !training.learning_rate.is_finite() || training.learning_rate <= 0.0 {
            return Err("Invalid learning rate".into());
        }

        if self.config.max_steps == Some(0) {
            return Err("Maximum steps must be greater than zero".into());
        }

        let total_steps = training
            .epochs
            .checked_mul(dataset.pairs.len())
            .ok_or("Total training steps overflow")?;

        let total_steps_u64 = u64::try_from(total_steps)
            .map_err(|_| "Total training steps exceed scheduler range")?;

        let (inputs, targets) = dataset.vocabulary();

        if inputs.is_empty() || targets.is_empty() {
            return Err("Training vocabulary is empty".into());
        }

        let input_ids: HashMap<&str, usize> = inputs
            .iter()
            .enumerate()
            .map(|(i, word)| (word.as_str(), i))
            .collect();

        let target_ids: HashMap<&str, usize> = targets
            .iter()
            .enumerate()
            .map(|(i, word)| (word.as_str(), i))
            .collect();

        let mut model = LearningModel::new(inputs.len(), training.hidden_size, targets.len())?;

        let sensors = TrainingSensors::default();
        let brain_controller = TrainingBrainController::new();
        let gradient_controller = GradientController::default();

        let optimizer_kind = OptimizerKind::from_keyword(optimizer_keyword)?;
        let optimizer_config = OptimizerConfig::new(optimizer_kind, training.learning_rate)?;

        let mut optimizer = OptimizerEngine::new(optimizer_config)?;

        let scheduler_kind = SchedulerKind::from_keyword(scheduler_keyword)?;

        let mut scheduler_config =
            SchedulerConfig::new(scheduler_kind, training.learning_rate, total_steps_u64)?;

        if scheduler_kind == SchedulerKind::WarmupCosine {
            scheduler_config.warmup_steps = (total_steps_u64 / 10).max(1).min(total_steps_u64);

            scheduler_config.validate()?;
        }

        let mut scheduler = LearningRateScheduler::new(scheduler_config)?;

        let mut steps = 0usize;
        let mut completed_epochs = 0usize;
        let mut initial_loss = 0.0f32;
        let mut final_loss = 0.0f32;
        let mut best_loss = f32::INFINITY;
        let mut warnings = 0usize;
        let mut critical_events = 0usize;

        // Adaptive multiplier remains independent of scheduler progression.
        let mut adaptive_lr_scale = 1.0f32;

        for epoch in 0..training.epochs {
            let mut epoch_loss = 0.0f32;

            for pair in &dataset.pairs {
                if self.config.max_steps == Some(steps) {
                    self.checkpoints
                        .save_latest(&model, &inputs, &targets, final_loss.max(0.0))?;

                    return Ok((
                        model,
                        TrainingReport {
                            status: TrainingStatus::PausedAtStepLimit,
                            steps,
                            completed_epochs,
                            initial_loss,
                            final_loss,
                            best_loss,
                            warnings,
                            critical_events,
                        },
                    ));
                }

                let input_id = *input_ids
                    .get(pair.input.as_str())
                    .ok_or("Missing input vocabulary")?;

                let target_id = *target_ids
                    .get(pair.target.as_str())
                    .ok_or("Missing target vocabulary")?;

                let scheduler_lr = scheduler.current_learning_rate();

                let effective_lr = scheduler_lr * adaptive_lr_scale;

                if !effective_lr.is_finite() || effective_lr <= 0.0 {
                    return Err("Invalid effective learning rate".into());
                }

                optimizer.config.learning_rate = effective_lr;

                let (diagnostics, gradient_report) =
                    model.calculate_gradients(input_id, target_id, &gradient_controller)?;

                let loss = diagnostics.loss;

                if gradient_report.exploded {
                    critical_events += 1;

                    self.checkpoints
                        .save_latest(&model, &inputs, &targets, loss)?;

                    return Ok((
                        model,
                        TrainingReport {
                            status: TrainingStatus::StoppedCritical,
                            steps,
                            completed_epochs,
                            initial_loss,
                            final_loss: loss,
                            best_loss,
                            warnings,
                            critical_events,
                        },
                    ));
                }

                let gradient_config = gradient_controller.config();

                optimizer.update(
                    &mut model,
                    input_id,
                    &diagnostics.gradients,
                    gradient_config.freeze_embeddings,
                    gradient_config.freeze_output_weights,
                    gradient_config.freeze_output_bias,
                )?;

                steps += 1;

                let (metrics, monitor_state) =
                    self.config
                        .monitor
                        .inspect(&model, loss, steps, epoch + 1)?;

                let snapshot = sensors.inspect(&model, &diagnostics, steps, epoch + 1)?;

                let decision = brain_controller.decide(&snapshot, monitor_state);

                match decision {
                    TrainingDecision::Normal => {}

                    TrainingDecision::Warning => {
                        warnings += 1;

                        // Reduce the adaptive multiplier relative to the
                        // current scheduler LR, not the original base LR.
                        let adjusted_lr = brain_controller.adjust_learning_rate(
                            scheduler.current_learning_rate(),
                            effective_lr,
                            TrainingDecision::Warning,
                        )?;

                        adaptive_lr_scale =
                            (adjusted_lr / scheduler.current_learning_rate()).clamp(0.0, 1.0);
                    }

                    TrainingDecision::Critical => {
                        critical_events += 1;

                        self.checkpoints
                            .save_latest(&model, &inputs, &targets, loss)?;

                        return Ok((
                            model,
                            TrainingReport {
                                status: TrainingStatus::StoppedCritical,
                                steps,
                                completed_epochs,
                                initial_loss,
                                final_loss: loss,
                                best_loss,
                                warnings,
                                critical_events,
                            },
                        ));
                    }
                }

                if !metrics.weight_norm.is_finite()
                    || !snapshot.gradients.l2_norm.is_finite()
                    || !snapshot.max_activation.is_finite()
                {
                    return Err("Training sensor detected non-finite metrics".into());
                }

                epoch_loss += loss;
                final_loss = loss;

                if steps == 1 {
                    initial_loss = loss;
                }

                // Advance scheduler only after the completed optimizer step.
                scheduler.advance()?;
            }

            completed_epochs += 1;

            let average_loss = epoch_loss / dataset.pairs.len() as f32;

            if average_loss < best_loss {
                best_loss = average_loss;
            }

            self.checkpoints
                .save_latest(&model, &inputs, &targets, average_loss)?;

            self.checkpoints
                .save_best_if_improved(&model, &inputs, &targets, average_loss)?;
        }

        Ok((
            model,
            TrainingReport {
                status: TrainingStatus::Completed,
                steps,
                completed_epochs,
                initial_loss,
                final_loss,
                best_loss,
                warnings,
                critical_events,
            },
        ))
    }
}
