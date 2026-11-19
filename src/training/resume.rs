use crate::learning::{
    CheckpointManager, GradientController, LearningDataset, LearningModel, LearningRateScheduler,
    OptimizerConfig, OptimizerEngine, OptimizerKind, OptimizerStateManager, SchedulerConfig,
    SchedulerKind, SchedulerStateManager, TrainingConfig, WeightMonitor,
};

use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

use super::monitor::{StabilityState, TrainingMonitor};

const MAGIC: &[u8; 8] = b"DAVRES12";

#[derive(Debug, Clone)]
pub struct ResumeState {
    pub epoch: usize,
    pub pair_cursor: usize,
    pub steps: usize,
    pub completed_epochs: usize,
    pub initial_loss: f32,
    pub final_loss: f32,
    pub best_loss: f32,
    pub epoch_loss: f32,
    pub warnings: usize,
    pub dataset_fingerprint: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResumeStatus {
    Completed,
    Paused,
    StoppedCritical,
}

#[derive(Debug, Clone)]
pub struct ResumeReport {
    pub status: ResumeStatus,
    pub steps: usize,
    pub completed_epochs: usize,
    pub resumed: bool,
    pub final_loss: f32,
}

pub struct ResumeManager {
    path: PathBuf,
}

impl ResumeManager {
    pub fn new(path: impl AsRef<Path>) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn exists(&self) -> bool {
        self.path.exists()
    }

    pub fn remove(&self) -> Result<(), String> {
        if self.path.exists() {
            fs::remove_file(&self.path).map_err(|e| format!("Resume removal error: {e}"))?;
        }

        Ok(())
    }

    pub fn save(
        &self,
        model: &LearningModel,
        inputs: &[String],
        targets: &[String],
        state: &ResumeState,
        config: &TrainingConfig,
    ) -> Result<(), String> {
        WeightMonitor::validate(model)?;

        if model.input_size != inputs.len() || model.output_size != targets.len() {
            return Err("Resume vocabulary mismatch".into());
        }

        if config.epochs == 0
            || config.hidden_size != model.hidden_size
            || !config.learning_rate.is_finite()
            || config.learning_rate <= 0.0
        {
            return Err("Resume configuration mismatch".into());
        }

        let mut bytes = Vec::new();

        bytes.extend_from_slice(MAGIC);

        for value in [
            model.input_size,
            model.hidden_size,
            model.output_size,
            state.epoch,
            state.pair_cursor,
            state.steps,
            state.completed_epochs,
            state.warnings,
            config.epochs,
            config.hidden_size,
        ] {
            bytes.extend_from_slice(&(value as u64).to_le_bytes());
        }

        for value in [
            config.learning_rate,
            state.initial_loss,
            state.final_loss,
            state.best_loss,
            state.epoch_loss,
        ] {
            if !value.is_finite() {
                return Err("Non-finite resume state".into());
            }

            bytes.extend_from_slice(&value.to_le_bytes());
        }

        bytes.extend_from_slice(&state.dataset_fingerprint.to_le_bytes());

        for vocabulary in [inputs, targets] {
            bytes.extend_from_slice(&(vocabulary.len() as u64).to_le_bytes());

            for word in vocabulary {
                let encoded = word.as_bytes();

                let length =
                    u32::try_from(encoded.len()).map_err(|_| "Vocabulary entry too large")?;

                bytes.extend_from_slice(&length.to_le_bytes());
                bytes.extend_from_slice(encoded);
            }
        }

        for weight in model
            .embeddings
            .iter()
            .flatten()
            .chain(model.output_weights.iter().flatten())
            .chain(model.output_bias.iter())
        {
            if !weight.is_finite() {
                return Err("Invalid resume weight".into());
            }

            bytes.extend_from_slice(&weight.to_le_bytes());
        }

        let checksum = Self::checksum(&bytes);

        bytes.extend_from_slice(&checksum.to_le_bytes());

        self.atomic_write(&bytes)
    }

    pub fn load(
        &self,
        dataset: &LearningDataset,
        config: &TrainingConfig,
    ) -> Result<(LearningModel, ResumeState), String> {
        let bytes = fs::read(&self.path).map_err(|e| format!("Resume read error: {e}"))?;

        if bytes.len() < 16 || &bytes[..8] != MAGIC {
            return Err("Invalid resume checkpoint".into());
        }

        let payload_end = bytes.len() - 8;

        let expected = u64::from_le_bytes(
            bytes[payload_end..]
                .try_into()
                .map_err(|_| "Invalid resume checksum")?,
        );

        let payload = &bytes[..payload_end];

        if Self::checksum(payload) != expected {
            return Err("Resume checksum mismatch".into());
        }

        let mut cursor = 8usize;

        let input_size = Self::read_u64(payload, &mut cursor)? as usize;
        let hidden_size = Self::read_u64(payload, &mut cursor)? as usize;
        let output_size = Self::read_u64(payload, &mut cursor)? as usize;

        let epoch = Self::read_u64(payload, &mut cursor)? as usize;
        let pair_cursor = Self::read_u64(payload, &mut cursor)? as usize;
        let steps = Self::read_u64(payload, &mut cursor)? as usize;
        let completed_epochs = Self::read_u64(payload, &mut cursor)? as usize;
        let warnings = Self::read_u64(payload, &mut cursor)? as usize;

        let saved_epochs = Self::read_u64(payload, &mut cursor)? as usize;
        let saved_hidden = Self::read_u64(payload, &mut cursor)? as usize;

        let learning_rate = Self::read_f32(payload, &mut cursor)?;
        let initial_loss = Self::read_f32(payload, &mut cursor)?;
        let final_loss = Self::read_f32(payload, &mut cursor)?;
        let best_loss = Self::read_f32(payload, &mut cursor)?;
        let epoch_loss = Self::read_f32(payload, &mut cursor)?;

        let fingerprint = Self::read_u64(payload, &mut cursor)?;

        if input_size == 0 || hidden_size == 0 || output_size == 0 {
            return Err("Invalid resume dimensions".into());
        }

        if saved_epochs != config.epochs
            || saved_hidden != config.hidden_size
            || learning_rate.to_bits() != config.learning_rate.to_bits()
        {
            return Err("Training configuration changed since checkpoint".into());
        }

        let (expected_inputs, expected_targets) = dataset.vocabulary();

        let inputs = Self::read_vocabulary(payload, &mut cursor)?;
        let targets = Self::read_vocabulary(payload, &mut cursor)?;

        if inputs != expected_inputs || targets != expected_targets {
            return Err("Training vocabulary changed".into());
        }

        if input_size != inputs.len()
            || output_size != targets.len()
            || hidden_size != config.hidden_size
        {
            return Err("Resume model dimensions mismatch".into());
        }

        if fingerprint != Self::dataset_fingerprint(dataset) {
            return Err("Training dataset sequence changed".into());
        }

        if epoch > config.epochs || pair_cursor > dataset.pairs.len() {
            return Err("Invalid resume cursor".into());
        }

        let mut model = LearningModel::new(input_size, hidden_size, output_size)?;

        for weight in model.embeddings.iter_mut().flatten() {
            *weight = Self::read_f32(payload, &mut cursor)?;
        }

        for weight in model.output_weights.iter_mut().flatten() {
            *weight = Self::read_f32(payload, &mut cursor)?;
        }

        for bias in &mut model.output_bias {
            *bias = Self::read_f32(payload, &mut cursor)?;
        }

        if cursor != payload.len() {
            return Err("Unexpected resume checkpoint data".into());
        }

        WeightMonitor::validate(&model)?;

        Ok((
            model,
            ResumeState {
                epoch,
                pair_cursor,
                steps,
                completed_epochs,
                initial_loss,
                final_loss,
                best_loss,
                epoch_loss,
                warnings,
                dataset_fingerprint: fingerprint,
            },
        ))
    }

    pub fn dataset_fingerprint(dataset: &LearningDataset) -> u64 {
        let mut hash = 14695981039346656037u64;

        for pair in &dataset.pairs {
            for byte in pair
                .input
                .as_bytes()
                .iter()
                .chain([0xFF].iter())
                .chain(pair.target.as_bytes())
                .chain([0xFE].iter())
            {
                hash ^= *byte as u64;
                hash = hash.wrapping_mul(1099511628211);
            }
        }

        hash
    }

    fn read_u64(bytes: &[u8], cursor: &mut usize) -> Result<u64, String> {
        let end = cursor.checked_add(8).ok_or("Cursor overflow")?;

        let value = bytes
            .get(*cursor..end)
            .ok_or("Truncated resume checkpoint")?;

        *cursor = end;

        Ok(u64::from_le_bytes(
            value.try_into().map_err(|_| "Invalid integer")?,
        ))
    }

    fn read_f32(bytes: &[u8], cursor: &mut usize) -> Result<f32, String> {
        let end = cursor.checked_add(4).ok_or("Cursor overflow")?;

        let value = bytes
            .get(*cursor..end)
            .ok_or("Truncated resume checkpoint")?;

        *cursor = end;

        let result = f32::from_le_bytes(value.try_into().map_err(|_| "Invalid float")?);

        if !result.is_finite() {
            return Err("Invalid resume float".into());
        }

        Ok(result)
    }

    fn read_vocabulary(bytes: &[u8], cursor: &mut usize) -> Result<Vec<String>, String> {
        let count = Self::read_u64(bytes, cursor)? as usize;

        if count > 1_000_000 {
            return Err("Vocabulary exceeds safety limit".into());
        }

        let mut words = Vec::with_capacity(count);

        for _ in 0..count {
            let end = cursor.checked_add(4).ok_or("Cursor overflow")?;

            let length_bytes = bytes.get(*cursor..end).ok_or("Truncated vocabulary")?;

            *cursor = end;

            let length =
                u32::from_le_bytes(length_bytes.try_into().map_err(|_| "Invalid length")?) as usize;

            let end = cursor.checked_add(length).ok_or("Cursor overflow")?;

            let word_bytes = bytes
                .get(*cursor..end)
                .ok_or("Truncated vocabulary entry")?;

            *cursor = end;

            let word =
                String::from_utf8(word_bytes.to_vec()).map_err(|_| "Invalid vocabulary UTF-8")?;

            words.push(word);
        }

        Ok(words)
    }

    fn checksum(bytes: &[u8]) -> u64 {
        let mut hash = 14695981039346656037u64;

        for byte in bytes {
            hash ^= *byte as u64;
            hash = hash.wrapping_mul(1099511628211);
        }

        hash
    }

    fn atomic_write(&self, bytes: &[u8]) -> Result<(), String> {
        let temp = self.path.with_extension("resume.tmp");

        let result = (|| {
            let mut file = File::create(&temp).map_err(|e| format!("Resume temp error: {e}"))?;

            file.write_all(bytes)
                .map_err(|e| format!("Resume write error: {e}"))?;

            file.sync_all()
                .map_err(|e| format!("Resume sync error: {e}"))?;

            drop(file);

            fs::rename(&temp, &self.path)
                .map_err(|e| format!("Resume atomic replace error: {e}"))?;

            Ok(())
        })();

        if result.is_err() {
            let _ = fs::remove_file(&temp);
        }

        result
    }
}

struct AdaptiveScaleManager {
    path: PathBuf,
}

impl AdaptiveScaleManager {
    const MAGIC: &'static [u8; 8] = b"DAVADP18";

    fn new(path: impl AsRef<Path>) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
        }
    }

    fn exists(&self) -> bool {
        self.path.exists()
    }

    fn checksum(bytes: &[u8]) -> u64 {
        let mut hash = 14695981039346656037u64;
        for byte in bytes {
            hash ^= *byte as u64;
            hash = hash.wrapping_mul(1099511628211);
        }
        hash
    }

    fn save(&self, scale: f32) -> Result<(), String> {
        if !scale.is_finite() || !(0.0..=1.0).contains(&scale) || scale == 0.0 {
            return Err("Invalid adaptive learning-rate scale".into());
        }

        let mut bytes = Vec::new();
        bytes.extend_from_slice(Self::MAGIC);
        bytes.extend_from_slice(&scale.to_le_bytes());

        let checksum = Self::checksum(&bytes);
        bytes.extend_from_slice(&checksum.to_le_bytes());

        let temp = self.path.with_extension("adaptive.tmp");

        let result = (|| {
            let mut file = File::create(&temp)
                .map_err(|e| format!("Adaptive checkpoint create error: {e}"))?;

            file.write_all(&bytes)
                .map_err(|e| format!("Adaptive checkpoint write error: {e}"))?;

            file.sync_all()
                .map_err(|e| format!("Adaptive checkpoint sync error: {e}"))?;

            drop(file);

            fs::rename(&temp, &self.path)
                .map_err(|e| format!("Adaptive checkpoint replace error: {e}"))?;

            Ok(())
        })();

        if result.is_err() {
            let _ = fs::remove_file(&temp);
        }

        result
    }

    fn load(&self) -> Result<f32, String> {
        let bytes =
            fs::read(&self.path).map_err(|e| format!("Adaptive checkpoint read error: {e}"))?;

        if bytes.len() != 20 || &bytes[..8] != Self::MAGIC {
            return Err("Invalid adaptive checkpoint".into());
        }

        let expected = u64::from_le_bytes(
            bytes[12..20]
                .try_into()
                .map_err(|_| "Invalid adaptive checksum")?,
        );

        if Self::checksum(&bytes[..12]) != expected {
            return Err("Adaptive checkpoint checksum mismatch".into());
        }

        let scale = f32::from_le_bytes(
            bytes[8..12]
                .try_into()
                .map_err(|_| "Invalid adaptive scale")?,
        );

        if !scale.is_finite() || scale <= 0.0 || scale > 1.0 {
            return Err("Invalid restored adaptive scale".into());
        }

        Ok(scale)
    }

    fn remove(&self) -> Result<(), String> {
        if self.path.exists() {
            fs::remove_file(&self.path)
                .map_err(|e| format!("Adaptive checkpoint removal error: {e}"))?;
        }
        Ok(())
    }
}

pub struct ResumableTrainer {
    pub resume: ResumeManager,
    pub checkpoints: CheckpointManager,
    pub monitor: TrainingMonitor,
}

impl ResumableTrainer {
    pub fn train(
        &self,
        dataset: &LearningDataset,
        config: &TrainingConfig,
        max_steps: Option<usize>,
    ) -> Result<(LearningModel, ResumeReport), String> {
        self.train_with_optimizer(dataset, config, max_steps, "sgd")
    }

    pub fn train_with_optimizer(
        &self,
        dataset: &LearningDataset,
        config: &TrainingConfig,
        max_steps: Option<usize>,
        optimizer_keyword: &str,
    ) -> Result<(LearningModel, ResumeReport), String> {
        self.train_with_optimizer_and_scheduler(
            dataset,
            config,
            max_steps,
            optimizer_keyword,
            "constant",
        )
    }

    pub fn train_with_optimizer_and_scheduler(
        &self,
        dataset: &LearningDataset,
        config: &TrainingConfig,
        max_steps: Option<usize>,
        optimizer_keyword: &str,
        scheduler_keyword: &str,
    ) -> Result<(LearningModel, ResumeReport), String> {
        if dataset.pairs.is_empty() {
            return Err("Training dataset is empty".into());
        }

        if config.epochs == 0
            || config.hidden_size == 0
            || !config.learning_rate.is_finite()
            || config.learning_rate <= 0.0
            || max_steps == Some(0)
        {
            return Err("Invalid resumable training configuration".into());
        }

        let (inputs, targets) = dataset.vocabulary();

        let resumed = self.resume.exists();

        let fingerprint = ResumeManager::dataset_fingerprint(dataset);

        let (mut model, mut state) = if resumed {
            self.resume.load(dataset, config)?
        } else {
            let model = LearningModel::new(inputs.len(), config.hidden_size, targets.len())?;

            let state = ResumeState {
                epoch: 0,
                pair_cursor: 0,
                steps: 0,
                completed_epochs: 0,
                initial_loss: 0.0,
                final_loss: 0.0,
                best_loss: f32::MAX,
                epoch_loss: 0.0,
                warnings: 0,
                dataset_fingerprint: fingerprint,
            };

            (model, state)
        };

        let starting_steps = state.steps;

        let adaptive_manager =
            AdaptiveScaleManager::new(self.resume.path().with_extension("adaptive"));

        let mut adaptive_lr_scale = if resumed {
            if !adaptive_manager.exists() {
                return Err("Adaptive LR checkpoint missing during resume".into());
            }
            adaptive_manager.load()?
        } else {
            1.0
        };

        let optimizer_kind = OptimizerKind::from_keyword(optimizer_keyword)?;
        let optimizer_config = OptimizerConfig::new(optimizer_kind, config.learning_rate)?;
        let optimizer_manager =
            OptimizerStateManager::new(self.resume.path().with_extension("optimizer"));

        let mut optimizer = if resumed {
            optimizer_manager.load(&model, state.steps)?
        } else {
            OptimizerEngine::new(optimizer_config)?
        };

        if optimizer.config.kind != optimizer_kind {
            return Err("Optimizer type changed since resume checkpoint".into());
        }

        let total_steps = (config.epochs as u64)
            .checked_mul(dataset.pairs.len() as u64)
            .ok_or("Scheduler total step overflow")?;

        let scheduler_kind = SchedulerKind::from_keyword(scheduler_keyword)?;
        let mut scheduler_config =
            SchedulerConfig::new(scheduler_kind, config.learning_rate, total_steps.max(1))?;

        if scheduler_kind == SchedulerKind::WarmupCosine {
            scheduler_config.warmup_steps = (total_steps / 10).max(1).min(total_steps);
        }

        let scheduler_manager =
            SchedulerStateManager::new(self.resume.path().with_extension("scheduler"));

        let mut scheduler = if resumed {
            if !scheduler_manager.exists() {
                return Err("Scheduler checkpoint missing during resume".into());
            }
            let restored = scheduler_manager.load()?;

            if restored.state.step != state.steps as u64 {
                return Err("Scheduler and training resume steps do not match".into());
            }

            if restored.config.kind != scheduler_kind
                || restored.config.base_learning_rate.to_bits() != config.learning_rate.to_bits()
                || restored.config.total_steps != total_steps.max(1)
            {
                return Err("Scheduler configuration changed since resume".into());
            }

            restored
        } else {
            LearningRateScheduler::new(scheduler_config)?
        };

        optimizer.config.learning_rate = scheduler.current_learning_rate() * adaptive_lr_scale;

        if state.epoch == config.epochs {
            return Ok((
                model,
                ResumeReport {
                    status: ResumeStatus::Completed,
                    steps: state.steps,
                    completed_epochs: state.completed_epochs,
                    resumed,
                    final_loss: state.final_loss,
                },
            ));
        }

        while state.epoch < config.epochs {
            while state.pair_cursor < dataset.pairs.len() {
                let pair = &dataset.pairs[state.pair_cursor];

                let input_id = inputs
                    .iter()
                    .position(|value| value == &pair.input)
                    .ok_or("Missing input vocabulary")?;

                let target_id = targets
                    .iter()
                    .position(|value| value == &pair.target)
                    .ok_or("Missing target vocabulary")?;

                let gradient_controller = GradientController::default();

                let (diagnostics, gradient_report) =
                    model.calculate_gradients(input_id, target_id, &gradient_controller)?;

                if gradient_report.exploded {
                    state.dataset_fingerprint = fingerprint;
                    self.save_state(
                        &model,
                        &inputs,
                        &targets,
                        &state,
                        config,
                        &optimizer,
                        &scheduler,
                        adaptive_lr_scale,
                    )?;

                    return Ok((
                        model,
                        ResumeReport {
                            status: ResumeStatus::StoppedCritical,
                            steps: state.steps,
                            completed_epochs: state.completed_epochs,
                            resumed,
                            final_loss: diagnostics.loss,
                        },
                    ));
                }

                let gradient_config = gradient_controller.config();

                optimizer.config.learning_rate =
                    scheduler.current_learning_rate() * adaptive_lr_scale;

                optimizer.update(
                    &mut model,
                    input_id,
                    &diagnostics.gradients,
                    gradient_config.freeze_embeddings,
                    gradient_config.freeze_output_weights,
                    gradient_config.freeze_output_bias,
                )?;

                let loss = diagnostics.loss;

                state.steps += 1;
                state.pair_cursor += 1;

                let next_rate = scheduler.advance()?;
                optimizer.config.learning_rate = next_rate * adaptive_lr_scale;
                state.final_loss = loss;
                state.epoch_loss += loss;

                let (_, stability) =
                    self.monitor
                        .inspect(&model, loss, state.steps, state.epoch + 1)?;

                match stability {
                    StabilityState::Normal => {}
                    StabilityState::Warning => {
                        state.warnings += 1;

                        let minimum_scale = 0.1f32;

                        adaptive_lr_scale = (adaptive_lr_scale * 0.5).max(minimum_scale);
                    }
                    StabilityState::Critical => {
                        state.dataset_fingerprint = fingerprint;

                        self.save_state(
                            &model,
                            &inputs,
                            &targets,
                            &state,
                            config,
                            &optimizer,
                            &scheduler,
                            adaptive_lr_scale,
                        )?;

                        return Ok((
                            model,
                            ResumeReport {
                                status: ResumeStatus::StoppedCritical,
                                steps: state.steps,
                                completed_epochs: state.completed_epochs,
                                resumed,
                                final_loss: loss,
                            },
                        ));
                    }
                }

                // Save the exact state after each completed sample.
                self.save_state(
                    &model,
                    &inputs,
                    &targets,
                    &state,
                    config,
                    &optimizer,
                    &scheduler,
                    adaptive_lr_scale,
                )?;

                if max_steps.map_or(false, |limit| state.steps - starting_steps >= limit) {
                    return Ok((
                        model,
                        ResumeReport {
                            status: ResumeStatus::Paused,
                            steps: state.steps,
                            completed_epochs: state.completed_epochs,
                            resumed,
                            final_loss: state.final_loss,
                        },
                    ));
                }
            }

            let average = state.epoch_loss / dataset.pairs.len() as f32;

            state.best_loss = state.best_loss.min(average);
            state.completed_epochs += 1;
            state.epoch += 1;
            state.pair_cursor = 0;
            state.epoch_loss = 0.0;

            self.checkpoints
                .save_latest(&model, &inputs, &targets, average)?;

            self.checkpoints
                .save_best_if_improved(&model, &inputs, &targets, average)?;

            self.save_state(
                &model,
                &inputs,
                &targets,
                &state,
                config,
                &optimizer,
                &scheduler,
                adaptive_lr_scale,
            )?;
        }

        self.resume.remove()?;
        optimizer_manager.remove()?;
        scheduler_manager.remove()?;
        adaptive_manager.remove()?;

        Ok((
            model,
            ResumeReport {
                status: ResumeStatus::Completed,
                steps: state.steps,
                completed_epochs: state.completed_epochs,
                resumed,
                final_loss: state.final_loss,
            },
        ))
    }

    fn save_state(
        &self,
        model: &LearningModel,
        inputs: &[String],
        targets: &[String],
        state: &ResumeState,
        config: &TrainingConfig,
        optimizer: &OptimizerEngine,
        scheduler: &LearningRateScheduler,
        adaptive_lr_scale: f32,
    ) -> Result<(), String> {
        self.checkpoints
            .save_latest(model, inputs, targets, state.final_loss.max(0.0))?;

        self.resume.save(model, inputs, targets, state, config)?;

        let manager = OptimizerStateManager::new(self.resume.path().with_extension("optimizer"));

        manager.save(optimizer, model, state.steps)?;

        let scheduler_manager =
            SchedulerStateManager::new(self.resume.path().with_extension("scheduler"));

        scheduler_manager.save(scheduler)?;

        let adaptive_manager =
            AdaptiveScaleManager::new(self.resume.path().with_extension("adaptive"));

        adaptive_manager.save(adaptive_lr_scale)?;

        if scheduler.state.step != state.steps as u64 {
            return Err("Scheduler and training step mismatch during save".into());
        }

        Ok(())
    }
}
