use crate::learning::LearningModel;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizerKind {
    Sgd,
    Momentum,
    Adam,
}

impl OptimizerKind {
    pub fn from_keyword(value: &str) -> Result<Self, String> {
        match value.trim().to_ascii_lowercase().as_str() {
            "sgd" => Ok(Self::Sgd),
            "momentum" => Ok(Self::Momentum),
            "adam" => Ok(Self::Adam),
            _ => Err(format!("Unsupported optimizer: {value}")),
        }
    }

    pub fn keyword(&self) -> &'static str {
        match self {
            Self::Sgd => "sgd",
            Self::Momentum => "momentum",
            Self::Adam => "adam",
        }
    }

    fn id(self) -> u8 {
        match self {
            Self::Sgd => 0,
            Self::Momentum => 1,
            Self::Adam => 2,
        }
    }

    fn from_id(id: u8) -> Result<Self, String> {
        match id {
            0 => Ok(Self::Sgd),
            1 => Ok(Self::Momentum),
            2 => Ok(Self::Adam),
            _ => Err("Invalid optimizer identifier".into()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct OptimizerConfig {
    pub kind: OptimizerKind,
    pub learning_rate: f32,
    pub momentum: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
}

impl OptimizerConfig {
    pub fn new(kind: OptimizerKind, learning_rate: f32) -> Result<Self, String> {
        let config = Self {
            kind,
            learning_rate,
            momentum: 0.9,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        };
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<(), String> {
        if !self.learning_rate.is_finite() || self.learning_rate <= 0.0 {
            return Err("Invalid optimizer learning rate".into());
        }

        if !self.momentum.is_finite()
            || !(0.0..1.0).contains(&self.momentum)
            || !self.beta1.is_finite()
            || !(0.0..1.0).contains(&self.beta1)
            || !self.beta2.is_finite()
            || !(0.0..1.0).contains(&self.beta2)
            || !self.epsilon.is_finite()
            || self.epsilon <= 0.0
        {
            return Err("Invalid optimizer hyperparameters".into());
        }

        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct OptimizerState {
    pub kind: OptimizerKind,
    pub step: u64,
    pub first_moment: Vec<f32>,
    pub second_moment: Vec<f32>,
}

impl OptimizerState {
    pub fn new(kind: OptimizerKind) -> Self {
        Self {
            kind,
            step: 0,
            first_moment: Vec::new(),
            second_moment: Vec::new(),
        }
    }
}

pub struct OptimizerEngine {
    pub config: OptimizerConfig,
    pub state: OptimizerState,
}

impl OptimizerEngine {
    pub fn new(config: OptimizerConfig) -> Result<Self, String> {
        config.validate()?;
        let state = OptimizerState::new(config.kind);
        Ok(Self { config, state })
    }

    pub fn from_state(config: OptimizerConfig, state: OptimizerState) -> Result<Self, String> {
        config.validate()?;

        if config.kind != state.kind {
            return Err("Optimizer configuration/state mismatch".into());
        }

        Ok(Self { config, state })
    }

    pub fn update(
        &mut self,
        model: &mut LearningModel,
        input_id: usize,
        gradients: &[f32],
        freeze_embeddings: bool,
        freeze_output_weights: bool,
        freeze_output_bias: bool,
    ) -> Result<(), String> {
        self.config.validate()?;

        if input_id >= model.input_size {
            return Err("Optimizer input ID out of range".into());
        }

        if self.state.kind != self.config.kind {
            return Err("Optimizer state kind mismatch".into());
        }

        let h = model.hidden_size;
        let o = model.output_size;

        let local_count = h + h * o + o;
        let global_count = model.input_size * h + h * o + o;

        if gradients.len() != local_count {
            return Err("Optimizer gradient dimension mismatch".into());
        }

        if gradients.iter().any(|x| !x.is_finite()) {
            return Err("Optimizer received non-finite gradients".into());
        }

        if self.state.first_moment.is_empty() {
            self.state.first_moment = vec![0.0; global_count];
        }

        if self.state.second_moment.is_empty() {
            self.state.second_moment = vec![0.0; global_count];
        }

        if self.state.first_moment.len() != global_count
            || self.state.second_moment.len() != global_count
        {
            return Err("Optimizer state dimension mismatch".into());
        }

        let mut parameters = Vec::with_capacity(global_count);
        parameters.extend(model.embeddings.iter().flatten().copied());
        parameters.extend(model.output_weights.iter().flatten().copied());
        parameters.extend(model.output_bias.iter().copied());

        if parameters.iter().any(|x| !x.is_finite()) {
            return Err("Non-finite model parameter".into());
        }

        let mut global_gradients = vec![0.0f32; global_count];

        let embedding_offset = input_id * h;

        for i in 0..h {
            global_gradients[embedding_offset + i] = gradients[i];
        }

        let weights_offset = model.input_size * h;

        for i in 0..(h * o) {
            global_gradients[weights_offset + i] = gradients[h + i];
        }

        let bias_offset = weights_offset + h * o;

        for i in 0..o {
            global_gradients[bias_offset + i] = gradients[h + h * o + i];
        }

        let next_step = self
            .state
            .step
            .checked_add(1)
            .ok_or("Optimizer step overflow")?;

        let mut updated = parameters.clone();

        for index in 0..global_count {
            let frozen = (index < weights_offset && freeze_embeddings)
                || (index >= weights_offset && index < bias_offset && freeze_output_weights)
                || (index >= bias_offset && freeze_output_bias);

            if frozen {
                continue;
            }

            let gradient = global_gradients[index];

            // Parameters that were not part of this sample have zero gradient.
            let delta = match self.config.kind {
                OptimizerKind::Sgd => gradient,

                OptimizerKind::Momentum => {
                    let velocity = self.config.momentum * self.state.first_moment[index] + gradient;

                    self.state.first_moment[index] = velocity;
                    velocity
                }

                OptimizerKind::Adam => {
                    let m = self.config.beta1 * self.state.first_moment[index]
                        + (1.0 - self.config.beta1) * gradient;

                    let v = self.config.beta2 * self.state.second_moment[index]
                        + (1.0 - self.config.beta2) * gradient * gradient;

                    self.state.first_moment[index] = m;
                    self.state.second_moment[index] = v;

                    let correction1 = 1.0 - self.config.beta1.powf(next_step as f32);

                    let correction2 = 1.0 - self.config.beta2.powf(next_step as f32);

                    let m_hat = m / correction1;
                    let v_hat = v / correction2;

                    m_hat / (v_hat.sqrt() + self.config.epsilon)
                }
            };

            updated[index] -= self.config.learning_rate * delta;
        }

        if updated.iter().any(|x| !x.is_finite())
            || self.state.first_moment.iter().any(|x| !x.is_finite())
            || self.state.second_moment.iter().any(|x| !x.is_finite())
        {
            return Err("Optimizer generated non-finite state".into());
        }

        let mut cursor = 0;

        for value in model.embeddings.iter_mut().flatten() {
            *value = updated[cursor];
            cursor += 1;
        }

        for value in model.output_weights.iter_mut().flatten() {
            *value = updated[cursor];
            cursor += 1;
        }

        for value in &mut model.output_bias {
            *value = updated[cursor];
            cursor += 1;
        }

        self.state.step = next_step;
        Ok(())
    }
}

pub struct OptimizerStateManager {
    path: PathBuf,
}

impl OptimizerStateManager {
    const MAGIC: &'static [u8; 8] = b"DAVOPT16";

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
            fs::remove_file(&self.path)
                .map_err(|e| format!("Optimizer state removal error: {e}"))?;
        }
        Ok(())
    }

    pub fn save(
        &self,
        engine: &OptimizerEngine,
        model: &LearningModel,
        resume_step: usize,
    ) -> Result<(), String> {
        engine.config.validate()?;

        if engine.state.step != resume_step as u64 {
            return Err("Optimizer/resume step mismatch during save".into());
        }

        let expected = model.input_size * model.hidden_size
            + model.hidden_size * model.output_size
            + model.output_size;

        if engine.state.first_moment.len() != expected
            || engine.state.second_moment.len() != expected
        {
            return Err("Optimizer state has not been initialized correctly".into());
        }

        let mut payload = Vec::new();

        payload.push(engine.config.kind.id());

        payload.extend_from_slice(&engine.state.step.to_le_bytes());

        for value in [
            engine.config.learning_rate,
            engine.config.momentum,
            engine.config.beta1,
            engine.config.beta2,
            engine.config.epsilon,
        ] {
            payload.extend_from_slice(&value.to_le_bytes());
        }

        for value in [
            model.input_size,
            model.hidden_size,
            model.output_size,
            expected,
        ] {
            payload.extend_from_slice(&(value as u64).to_le_bytes());
        }

        for value in engine
            .state
            .first_moment
            .iter()
            .chain(engine.state.second_moment.iter())
        {
            if !value.is_finite() {
                return Err("Non-finite optimizer state".into());
            }
            payload.extend_from_slice(&value.to_le_bytes());
        }

        let checksum = Self::checksum(&payload);

        let mut bytes = Vec::new();
        bytes.extend_from_slice(Self::MAGIC);
        bytes.extend_from_slice(&checksum.to_le_bytes());
        bytes.extend_from_slice(&payload);

        self.atomic_write(&bytes)
    }

    pub fn load(
        &self,
        model: &LearningModel,
        resume_step: usize,
    ) -> Result<OptimizerEngine, String> {
        let bytes = fs::read(&self.path).map_err(|e| format!("Optimizer state read error: {e}"))?;

        if bytes.len() < 17 || &bytes[..8] != Self::MAGIC {
            return Err("Invalid optimizer state header".into());
        }

        let expected_checksum =
            u64::from_le_bytes(bytes[8..16].try_into().map_err(|_| "Invalid checksum")?);

        let payload = &bytes[16..];

        if Self::checksum(payload) != expected_checksum {
            return Err("Optimizer state checksum mismatch".into());
        }

        let mut cursor = 0usize;

        let kind = OptimizerKind::from_id(Self::read_u8(payload, &mut cursor)?)?;
        let step = Self::read_u64(payload, &mut cursor)?;

        let learning_rate = Self::read_f32(payload, &mut cursor)?;
        let momentum = Self::read_f32(payload, &mut cursor)?;
        let beta1 = Self::read_f32(payload, &mut cursor)?;
        let beta2 = Self::read_f32(payload, &mut cursor)?;
        let epsilon = Self::read_f32(payload, &mut cursor)?;

        let input_size = Self::read_u64(payload, &mut cursor)? as usize;
        let hidden_size = Self::read_u64(payload, &mut cursor)? as usize;
        let output_size = Self::read_u64(payload, &mut cursor)? as usize;
        let count = Self::read_u64(payload, &mut cursor)? as usize;

        if input_size != model.input_size
            || hidden_size != model.hidden_size
            || output_size != model.output_size
        {
            return Err("Optimizer/model dimensions mismatch".into());
        }

        if step != resume_step as u64 {
            return Err("Optimizer/resume step mismatch".into());
        }

        let expected_count = input_size * hidden_size + hidden_size * output_size + output_size;

        if count != expected_count {
            return Err("Invalid optimizer state length".into());
        }

        let mut first = Vec::with_capacity(count);
        let mut second = Vec::with_capacity(count);

        for _ in 0..count {
            first.push(Self::read_f32(payload, &mut cursor)?);
        }

        for _ in 0..count {
            second.push(Self::read_f32(payload, &mut cursor)?);
        }

        if cursor != payload.len() {
            return Err("Unexpected optimizer state data".into());
        }

        let config = OptimizerConfig {
            kind,
            learning_rate,
            momentum,
            beta1,
            beta2,
            epsilon,
        };

        let state = OptimizerState {
            kind,
            step,
            first_moment: first,
            second_moment: second,
        };

        OptimizerEngine::from_state(config, state)
    }

    fn read_u8(bytes: &[u8], cursor: &mut usize) -> Result<u8, String> {
        let value = *bytes.get(*cursor).ok_or("Truncated optimizer state")?;
        *cursor += 1;
        Ok(value)
    }

    fn read_u64(bytes: &[u8], cursor: &mut usize) -> Result<u64, String> {
        let end = cursor.checked_add(8).ok_or("Cursor overflow")?;
        let value = bytes.get(*cursor..end).ok_or("Truncated optimizer state")?;
        *cursor = end;
        Ok(u64::from_le_bytes(
            value.try_into().map_err(|_| "Invalid integer")?,
        ))
    }

    fn read_f32(bytes: &[u8], cursor: &mut usize) -> Result<f32, String> {
        let end = cursor.checked_add(4).ok_or("Cursor overflow")?;
        let value = bytes.get(*cursor..end).ok_or("Truncated optimizer state")?;
        *cursor = end;

        let result = f32::from_le_bytes(value.try_into().map_err(|_| "Invalid float")?);

        if !result.is_finite() {
            return Err("Non-finite optimizer state value".into());
        }

        Ok(result)
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
        let temp = self.path.with_extension("optimizer.tmp");

        let result = (|| {
            let mut file = File::create(&temp).map_err(|e| format!("Optimizer temp error: {e}"))?;

            file.write_all(bytes)
                .map_err(|e| format!("Optimizer write error: {e}"))?;

            file.sync_all()
                .map_err(|e| format!("Optimizer sync error: {e}"))?;

            drop(file);

            fs::rename(&temp, &self.path)
                .map_err(|e| format!("Optimizer atomic replace error: {e}"))?;

            Ok(())
        })();

        if result.is_err() {
            let _ = fs::remove_file(&temp);
        }

        result
    }
}
