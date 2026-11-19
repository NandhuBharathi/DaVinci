use std::collections::BTreeSet;
use std::fs;

#[derive(Clone, Debug)]
pub struct LearningPair {
    pub input: String,
    pub target: String,
}

pub struct LearningDataset {
    pub pairs: Vec<LearningPair>,
}

impl LearningDataset {
    pub fn load(path: &str) -> Result<Self, String> {
        let content = fs::read_to_string(path).map_err(|e| format!("Dataset read error: {e}"))?;

        let mut pairs = Vec::new();
        let mut inside_guidance = false;

        for line in content.lines() {
            let line = line.trim();

            if line == "WORD_GUIDANCE:" {
                inside_guidance = true;
                continue;
            }

            if line == "RELATIONSHIPS:" || line == "END_SAMPLE" {
                inside_guidance = false;
                continue;
            }

            if !inside_guidance || line.is_empty() {
                continue;
            }

            let (input, target) = line
                .split_once('\t')
                .ok_or_else(|| format!("Invalid guidance row: {line}"))?;

            let input = input.trim();
            let target = target.trim();

            if input.is_empty() || target.is_empty() {
                return Err("Empty learning pair".into());
            }

            pairs.push(LearningPair {
                input: input.to_string(),
                target: target.to_string(),
            });
        }

        if pairs.is_empty() {
            return Err("No learning pairs found".into());
        }

        Ok(Self { pairs })
    }

    pub fn vocabulary(&self) -> (Vec<String>, Vec<String>) {
        let inputs: BTreeSet<String> = self.pairs.iter().map(|p| p.input.clone()).collect();

        let targets: BTreeSet<String> = self.pairs.iter().map(|p| p.target.clone()).collect();

        (inputs.into_iter().collect(), targets.into_iter().collect())
    }
}
