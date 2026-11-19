use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulerKind {
    Constant,
    StepDecay,
    ExponentialDecay,
    CosineAnnealing,
    WarmupCosine,
}

impl SchedulerKind {
    pub fn from_keyword(value: &str) -> Result<Self, String> {
        match value.trim().to_ascii_lowercase().as_str() {
            "constant" => Ok(Self::Constant),
            "step" | "step_decay" => Ok(Self::StepDecay),
            "exponential" | "exponential_decay" => Ok(Self::ExponentialDecay),
            "cosine" | "cosine_annealing" => Ok(Self::CosineAnnealing),
            "warmup_cosine" | "warmup" => Ok(Self::WarmupCosine),
            _ => Err(format!("Unknown scheduler: {value}")),
        }
    }

    fn id(self) -> u8 {
        match self {
            Self::Constant => 0,
            Self::StepDecay => 1,
            Self::ExponentialDecay => 2,
            Self::CosineAnnealing => 3,
            Self::WarmupCosine => 4,
        }
    }

    fn from_id(id: u8) -> Result<Self, String> {
        match id {
            0 => Ok(Self::Constant),
            1 => Ok(Self::StepDecay),
            2 => Ok(Self::ExponentialDecay),
            3 => Ok(Self::CosineAnnealing),
            4 => Ok(Self::WarmupCosine),
            _ => Err("Invalid scheduler identifier".into()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    pub kind: SchedulerKind,
    pub base_learning_rate: f32,
    pub minimum_learning_rate: f32,
    pub decay_factor: f32,
    pub step_size: u64,
    pub total_steps: u64,
    pub warmup_steps: u64,
}

impl SchedulerConfig {
    pub fn new(
        kind: SchedulerKind,
        base_learning_rate: f32,
        total_steps: u64,
    ) -> Result<Self, String> {
        let config = Self {
            kind,
            base_learning_rate,
            minimum_learning_rate: base_learning_rate * 0.01,
            decay_factor: 0.5,
            step_size: 100,
            total_steps,
            warmup_steps: 0,
        };

        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<(), String> {
        if !self.base_learning_rate.is_finite() || self.base_learning_rate <= 0.0 {
            return Err("Invalid base learning rate".into());
        }

        if !self.minimum_learning_rate.is_finite()
            || self.minimum_learning_rate < 0.0
            || self.minimum_learning_rate > self.base_learning_rate
        {
            return Err("Invalid minimum learning rate".into());
        }

        if !self.decay_factor.is_finite() || self.decay_factor <= 0.0 || self.decay_factor > 1.0 {
            return Err("Invalid decay factor".into());
        }

        if self.total_steps == 0 || self.step_size == 0 {
            return Err("Scheduler steps must be non-zero".into());
        }

        if self.warmup_steps > self.total_steps {
            return Err("Warmup cannot exceed total steps".into());
        }

        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct SchedulerState {
    pub step: u64,
    pub current_learning_rate: f32,
}

pub struct LearningRateScheduler {
    pub config: SchedulerConfig,
    pub state: SchedulerState,
}

impl LearningRateScheduler {
    pub fn new(config: SchedulerConfig) -> Result<Self, String> {
        config.validate()?;

        let initial = if config.kind == SchedulerKind::WarmupCosine && config.warmup_steps > 0 {
            config.base_learning_rate / config.warmup_steps as f32
        } else {
            config.base_learning_rate
        };

        Ok(Self {
            config,
            state: SchedulerState {
                step: 0,
                current_learning_rate: initial,
            },
        })
    }

    pub fn from_state(config: SchedulerConfig, state: SchedulerState) -> Result<Self, String> {
        config.validate()?;

        if state.step > config.total_steps
            || !state.current_learning_rate.is_finite()
            || state.current_learning_rate < 0.0
        {
            return Err("Invalid scheduler state".into());
        }

        let mut scheduler = Self::new(config)?;
        scheduler.state = state;

        let expected = scheduler.calculate_learning_rate(scheduler.state.step);

        if (expected - scheduler.state.current_learning_rate).abs() > 1e-6 {
            return Err("Scheduler state learning rate mismatch".into());
        }

        Ok(scheduler)
    }

    pub fn current_learning_rate(&self) -> f32 {
        self.state.current_learning_rate
    }

    pub fn advance(&mut self) -> Result<f32, String> {
        if self.state.step >= self.config.total_steps {
            return Ok(self.state.current_learning_rate);
        }

        self.state.step += 1;

        let learning_rate = self.calculate_learning_rate(self.state.step);

        if !learning_rate.is_finite() || learning_rate < 0.0 {
            return Err("Scheduler generated invalid learning rate".into());
        }

        self.state.current_learning_rate = learning_rate;

        Ok(learning_rate)
    }

    fn calculate_learning_rate(&self, step: u64) -> f32 {
        let base = self.config.base_learning_rate;
        let minimum = self.config.minimum_learning_rate;

        match self.config.kind {
            SchedulerKind::Constant => base,

            SchedulerKind::StepDecay => {
                let exponent = step / self.config.step_size;
                (base * self.config.decay_factor.powf(exponent as f32)).max(minimum)
            }

            SchedulerKind::ExponentialDecay => {
                (base * self.config.decay_factor.powf(step as f32)).max(minimum)
            }

            SchedulerKind::CosineAnnealing => {
                let progress =
                    (step.min(self.config.total_steps) as f32) / self.config.total_steps as f32;

                let cosine = (std::f32::consts::PI * progress).cos();

                minimum + 0.5 * (base - minimum) * (1.0 + cosine)
            }

            SchedulerKind::WarmupCosine => {
                if self.config.warmup_steps > 0 && step <= self.config.warmup_steps {
                    return base * step as f32 / self.config.warmup_steps as f32;
                }

                let decay_steps = self.config.total_steps - self.config.warmup_steps;

                if decay_steps == 0 {
                    return base;
                }

                let elapsed = step.saturating_sub(self.config.warmup_steps);

                let progress = (elapsed.min(decay_steps) as f32) / decay_steps as f32;

                let cosine = (std::f32::consts::PI * progress).cos();

                minimum + 0.5 * (base - minimum) * (1.0 + cosine)
            }
        }
    }
}

pub struct SchedulerStateManager {
    path: PathBuf,
}

impl SchedulerStateManager {
    const MAGIC: &'static [u8; 8] = b"DAVSCH17";

    pub fn new(path: impl AsRef<Path>) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
        }
    }

    pub fn exists(&self) -> bool {
        self.path.exists()
    }

    pub fn remove(&self) -> Result<(), String> {
        if self.path.exists() {
            fs::remove_file(&self.path)
                .map_err(|e| format!("Scheduler state removal error: {e}"))?;
        }
        Ok(())
    }

    pub fn save(&self, scheduler: &LearningRateScheduler) -> Result<(), String> {
        scheduler.config.validate()?;

        let mut payload = Vec::new();

        payload.push(scheduler.config.kind.id());

        payload.extend_from_slice(&scheduler.state.step.to_le_bytes());

        for value in [
            scheduler.config.base_learning_rate,
            scheduler.config.minimum_learning_rate,
            scheduler.config.decay_factor,
            scheduler.state.current_learning_rate,
        ] {
            payload.extend_from_slice(&value.to_le_bytes());
        }

        for value in [
            scheduler.config.step_size,
            scheduler.config.total_steps,
            scheduler.config.warmup_steps,
        ] {
            payload.extend_from_slice(&value.to_le_bytes());
        }

        let checksum = Self::checksum(&payload);

        let mut bytes = Vec::new();
        bytes.extend_from_slice(Self::MAGIC);
        bytes.extend_from_slice(&checksum.to_le_bytes());
        bytes.extend_from_slice(&payload);

        self.atomic_write(&bytes)
    }

    pub fn load(&self) -> Result<LearningRateScheduler, String> {
        let bytes = fs::read(&self.path).map_err(|e| format!("Scheduler state read error: {e}"))?;

        if bytes.len() != 65 || &bytes[..8] != Self::MAGIC {
            return Err("Invalid scheduler checkpoint".into());
        }

        let checksum = u64::from_le_bytes(bytes[8..16].try_into().map_err(|_| "Invalid checksum")?);

        let payload = &bytes[16..];

        if Self::checksum(payload) != checksum {
            return Err("Scheduler checksum mismatch".into());
        }

        let mut cursor = 0usize;

        let kind = SchedulerKind::from_id(Self::read_u8(payload, &mut cursor)?)?;
        let step = Self::read_u64(payload, &mut cursor)?;

        let base_learning_rate = Self::read_f32(payload, &mut cursor)?;
        let minimum_learning_rate = Self::read_f32(payload, &mut cursor)?;
        let decay_factor = Self::read_f32(payload, &mut cursor)?;
        let current_learning_rate = Self::read_f32(payload, &mut cursor)?;

        let step_size = Self::read_u64(payload, &mut cursor)?;
        let total_steps = Self::read_u64(payload, &mut cursor)?;
        let warmup_steps = Self::read_u64(payload, &mut cursor)?;

        if cursor != payload.len() {
            return Err("Unexpected scheduler checkpoint data".into());
        }

        let config = SchedulerConfig {
            kind,
            base_learning_rate,
            minimum_learning_rate,
            decay_factor,
            step_size,
            total_steps,
            warmup_steps,
        };

        LearningRateScheduler::from_state(
            config,
            SchedulerState {
                step,
                current_learning_rate,
            },
        )
    }

    fn read_u8(bytes: &[u8], cursor: &mut usize) -> Result<u8, String> {
        let value = *bytes.get(*cursor).ok_or("Truncated scheduler state")?;
        *cursor += 1;
        Ok(value)
    }

    fn read_u64(bytes: &[u8], cursor: &mut usize) -> Result<u64, String> {
        let end = cursor.checked_add(8).ok_or("Cursor overflow")?;
        let value = bytes.get(*cursor..end).ok_or("Truncated scheduler state")?;
        *cursor = end;

        Ok(u64::from_le_bytes(
            value.try_into().map_err(|_| "Invalid integer")?,
        ))
    }

    fn read_f32(bytes: &[u8], cursor: &mut usize) -> Result<f32, String> {
        let end = cursor.checked_add(4).ok_or("Cursor overflow")?;
        let value = bytes.get(*cursor..end).ok_or("Truncated scheduler state")?;
        *cursor = end;

        let result = f32::from_le_bytes(value.try_into().map_err(|_| "Invalid float")?);

        if !result.is_finite() {
            return Err("Invalid scheduler float".into());
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
        let temp = self.path.with_extension("scheduler.tmp");

        let result = (|| {
            let mut file = File::create(&temp).map_err(|e| format!("Scheduler temp error: {e}"))?;

            file.write_all(bytes)
                .map_err(|e| format!("Scheduler write error: {e}"))?;

            file.sync_all()
                .map_err(|e| format!("Scheduler sync error: {e}"))?;

            drop(file);

            fs::rename(&temp, &self.path)
                .map_err(|e| format!("Scheduler atomic replace error: {e}"))?;

            Ok(())
        })();

        if result.is_err() {
            let _ = fs::remove_file(&temp);
        }

        result
    }
}
