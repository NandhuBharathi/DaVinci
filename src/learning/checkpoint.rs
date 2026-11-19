use crate::learning::weights::WeightMonitor;
use crate::learning::LearningModel;

use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

const MAGIC: &[u8; 8] = b"DAVINC10";

#[derive(Debug, Clone)]
pub struct LoadedCheckpoint {
    pub model: LearningModel,
    pub input_vocabulary: Vec<String>,
    pub target_vocabulary: Vec<String>,
    pub validation_loss: f32,
}

pub struct CheckpointManager {
    directory: PathBuf,
}

impl CheckpointManager {
    pub fn new(directory: impl AsRef<Path>) -> Result<Self, String> {
        fs::create_dir_all(directory.as_ref())
            .map_err(|e| format!("Checkpoint directory error: {e}"))?;

        Ok(Self {
            directory: directory.as_ref().to_path_buf(),
        })
    }

    pub fn latest_path(&self) -> PathBuf {
        self.directory.join("latest.dvckpt")
    }

    pub fn best_path(&self) -> PathBuf {
        self.directory.join("best.dvckpt")
    }

    pub fn save_latest(
        &self,
        model: &LearningModel,
        inputs: &[String],
        targets: &[String],
        loss: f32,
    ) -> Result<(), String> {
        Self::validate(model, inputs, targets, loss)?;

        let bytes = Self::encode(model, inputs, targets, loss)?;

        Self::atomic_write(&self.latest_path(), &bytes)
    }

    pub fn save_best_if_improved(
        &self,
        model: &LearningModel,
        inputs: &[String],
        targets: &[String],
        validation_loss: f32,
    ) -> Result<bool, String> {
        Self::validate(model, inputs, targets, validation_loss)?;

        let best_path = self.best_path();

        if best_path.exists() {
            let previous = Self::load_file(&best_path)?;

            if validation_loss >= previous.validation_loss {
                return Ok(false);
            }
        }

        let bytes = Self::encode(model, inputs, targets, validation_loss)?;

        Self::atomic_write(&best_path, &bytes)?;

        Ok(true)
    }

    pub fn load_latest(&self) -> Result<LoadedCheckpoint, String> {
        Self::load_file(&self.latest_path())
    }

    pub fn load_best(&self) -> Result<LoadedCheckpoint, String> {
        Self::load_file(&self.best_path())
    }

    fn validate(
        model: &LearningModel,
        inputs: &[String],
        targets: &[String],
        loss: f32,
    ) -> Result<(), String> {
        WeightMonitor::validate(model)?;

        if model.input_size != inputs.len() || model.output_size != targets.len() {
            return Err("Checkpoint vocabulary mismatch".into());
        }

        if !loss.is_finite() || loss < 0.0 {
            return Err("Invalid checkpoint loss".into());
        }

        if inputs.iter().any(|s| s.trim().is_empty()) || targets.iter().any(|s| s.trim().is_empty())
        {
            return Err("Empty vocabulary entry".into());
        }

        Ok(())
    }

    fn encode(
        model: &LearningModel,
        inputs: &[String],
        targets: &[String],
        loss: f32,
    ) -> Result<Vec<u8>, String> {
        let mut payload = Vec::new();

        for dimension in [model.input_size, model.hidden_size, model.output_size] {
            payload.extend_from_slice(&(dimension as u64).to_le_bytes());
        }

        payload.extend_from_slice(&loss.to_le_bytes());

        for vocabulary in [inputs, targets] {
            payload.extend_from_slice(&(vocabulary.len() as u64).to_le_bytes());

            for word in vocabulary {
                let bytes = word.as_bytes();

                let length =
                    u32::try_from(bytes.len()).map_err(|_| "Vocabulary entry too large")?;

                payload.extend_from_slice(&length.to_le_bytes());
                payload.extend_from_slice(bytes);
            }
        }

        for value in model
            .embeddings
            .iter()
            .flatten()
            .chain(model.output_weights.iter().flatten())
            .chain(model.output_bias.iter())
        {
            payload.extend_from_slice(&value.to_le_bytes());
        }

        let checksum = Self::checksum(&payload);

        let mut result = Vec::new();

        result.extend_from_slice(MAGIC);
        result.extend_from_slice(&checksum.to_le_bytes());
        result.extend_from_slice(&payload);

        Ok(result)
    }

    fn decode(bytes: &[u8]) -> Result<LoadedCheckpoint, String> {
        if bytes.len() < 16 || &bytes[..8] != MAGIC {
            return Err("Invalid checkpoint header".into());
        }

        let expected = u64::from_le_bytes(bytes[8..16].try_into().map_err(|_| "Invalid checksum")?);

        let payload = &bytes[16..];

        if Self::checksum(payload) != expected {
            return Err("Checkpoint checksum mismatch".into());
        }

        let mut cursor = 0usize;

        let input_size = Self::read_u64(payload, &mut cursor)? as usize;
        let hidden_size = Self::read_u64(payload, &mut cursor)? as usize;
        let output_size = Self::read_u64(payload, &mut cursor)? as usize;

        if input_size == 0 || hidden_size == 0 || output_size == 0 {
            return Err("Invalid checkpoint dimensions".into());
        }

        let w1 = input_size
            .checked_mul(hidden_size)
            .ok_or("Checkpoint dimension overflow")?;

        let w2 = hidden_size
            .checked_mul(output_size)
            .ok_or("Checkpoint dimension overflow")?;

        let loss = Self::read_f32(payload, &mut cursor)?;

        if !loss.is_finite() || loss < 0.0 {
            return Err("Invalid checkpoint loss".into());
        }

        let inputs = Self::read_vocabulary(payload, &mut cursor, input_size)?;
        let targets = Self::read_vocabulary(payload, &mut cursor, output_size)?;

        let mut model = LearningModel::new(input_size, hidden_size, output_size)?;

        for value in model.embeddings.iter_mut().flatten() {
            *value = Self::read_f32(payload, &mut cursor)?;
        }

        for value in model.output_weights.iter_mut().flatten() {
            *value = Self::read_f32(payload, &mut cursor)?;
        }

        for value in &mut model.output_bias {
            *value = Self::read_f32(payload, &mut cursor)?;
        }

        if cursor != payload.len() {
            return Err("Unexpected checkpoint data".into());
        }

        if model.embeddings.iter().flatten().count() != w1
            || model.output_weights.iter().flatten().count() != w2
        {
            return Err("Checkpoint weight count mismatch".into());
        }

        WeightMonitor::validate(&model)?;

        Ok(LoadedCheckpoint {
            model,
            input_vocabulary: inputs,
            target_vocabulary: targets,
            validation_loss: loss,
        })
    }

    fn read_u64(bytes: &[u8], cursor: &mut usize) -> Result<u64, String> {
        let end = cursor.checked_add(8).ok_or("Cursor overflow")?;

        let slice = bytes.get(*cursor..end).ok_or("Truncated checkpoint")?;

        *cursor = end;

        Ok(u64::from_le_bytes(
            slice.try_into().map_err(|_| "Invalid integer")?,
        ))
    }

    fn read_f32(bytes: &[u8], cursor: &mut usize) -> Result<f32, String> {
        let end = cursor.checked_add(4).ok_or("Cursor overflow")?;

        let slice = bytes.get(*cursor..end).ok_or("Truncated checkpoint")?;

        *cursor = end;

        let value = f32::from_le_bytes(slice.try_into().map_err(|_| "Invalid float")?);

        if !value.is_finite() {
            return Err("Non-finite checkpoint value".into());
        }

        Ok(value)
    }

    fn read_vocabulary(
        bytes: &[u8],
        cursor: &mut usize,
        expected: usize,
    ) -> Result<Vec<String>, String> {
        let count = Self::read_u64(bytes, cursor)?;

        if count != expected as u64 {
            return Err("Vocabulary size mismatch".into());
        }

        let mut result = Vec::with_capacity(expected);

        for _ in 0..expected {
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
                String::from_utf8(word_bytes.to_vec()).map_err(|_| "Invalid UTF-8 vocabulary")?;

            if word.trim().is_empty() {
                return Err("Empty vocabulary entry".into());
            }

            result.push(word);
        }

        if result
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len()
            != result.len()
        {
            return Err("Duplicate vocabulary entries".into());
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

    fn atomic_write(path: &Path, bytes: &[u8]) -> Result<(), String> {
        let counter = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);

        let temp_path = path.with_extension(format!("tmp.{}.{}", std::process::id(), counter));

        let result = (|| {
            let mut file =
                File::create(&temp_path).map_err(|e| format!("Temporary checkpoint error: {e}"))?;

            file.write_all(bytes)
                .map_err(|e| format!("Checkpoint write error: {e}"))?;

            file.sync_all()
                .map_err(|e| format!("Checkpoint sync error: {e}"))?;

            drop(file);

            fs::rename(&temp_path, path)
                .map_err(|e| format!("Atomic checkpoint replace error: {e}"))?;

            Ok(())
        })();

        if result.is_err() {
            let _ = fs::remove_file(&temp_path);
        }

        result
    }

    fn load_file(path: &Path) -> Result<LoadedCheckpoint, String> {
        let mut file = File::open(path).map_err(|e| format!("Checkpoint open error: {e}"))?;

        let mut bytes = Vec::new();

        file.read_to_end(&mut bytes)
            .map_err(|e| format!("Checkpoint read error: {e}"))?;

        Self::decode(&bytes)
    }
}
