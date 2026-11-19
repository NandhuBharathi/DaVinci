use super::guidance::{GuidanceDataset, RelationGuidance, TrainingExample, WordGuidance};

pub struct GuidanceParser;

impl GuidanceParser {
    pub fn parse(input: &str) -> Result<GuidanceDataset, String> {
        let mut dataset = GuidanceDataset::new();

        let mut id = String::new();
        let mut content = String::new();
        let mut context = String::new();

        let mut words = Vec::new();
        let mut relationships = Vec::new();

        let mut section = String::new();
        let mut inside_sample = false;

        for (line_number, raw) in input.lines().enumerate() {
            let line = raw.trim();

            if line.is_empty() {
                continue;
            }

            if line == "SAMPLE" {
                if inside_sample {
                    return Err(format!("Nested SAMPLE at line {}", line_number + 1));
                }

                inside_sample = true;
                continue;
            }

            if !inside_sample {
                return Err(format!(
                    "Content outside SAMPLE at line {}",
                    line_number + 1
                ));
            }

            if let Some(value) = line.strip_prefix("ID:") {
                id = value.trim().to_string();
                continue;
            }

            if let Some(value) = line.strip_prefix("CONTENT:") {
                content = value.trim().to_string();
                section = "CONTENT".into();
                continue;
            }

            if line == "WORD_GUIDANCE:" {
                section = "WORD".into();
                continue;
            }

            if line == "RELATIONSHIPS:" {
                section = "RELATION".into();
                continue;
            }

            if let Some(value) = line.strip_prefix("CONTEXT:") {
                context = value.trim().to_string();
                section = "CONTEXT".into();
                continue;
            }

            if line == "END_SAMPLE" {
                if id.is_empty() || content.is_empty() {
                    return Err(format!(
                        "Sample missing ID or CONTENT at line {}",
                        line_number + 1
                    ));
                }

                dataset.add(TrainingExample {
                    id: id.clone(),
                    content: content.clone(),
                    word_guidance: words.clone(),
                    relationships: relationships.clone(),
                    context_instruction: context.clone(),
                });

                id.clear();
                content.clear();
                context.clear();
                words.clear();
                relationships.clear();
                section.clear();
                inside_sample = false;

                continue;
            }

            match section.as_str() {
                "WORD" => {
                    let parts: Vec<&str> = line.splitn(2, '\t').collect();

                    if parts.len() != 2 {
                        return Err(format!("Invalid word guidance at line {}", line_number + 1));
                    }

                    words.push(WordGuidance {
                        word: parts[0].trim().to_string(),
                        meaning_instruction: parts[1].trim().to_string(),
                    });
                }

                "RELATION" => {
                    let parts: Vec<&str> = line.split('\t').collect();

                    if parts.len() != 3 {
                        return Err(format!("Invalid relationship at line {}", line_number + 1));
                    }

                    relationships.push(RelationGuidance {
                        source: parts[0].trim().to_string(),
                        relation: parts[1].trim().to_string(),
                        target: parts[2].trim().to_string(),
                    });
                }

                "CONTENT" => {
                    content.push(' ');
                    content.push_str(line);
                }

                "CONTEXT" => {
                    context = line.to_string();
                }

                _ => {
                    return Err(format!(
                        "Unknown dataset section at line {}",
                        line_number + 1
                    ));
                }
            }
        }

        if inside_sample {
            return Err("Dataset ended before END_SAMPLE".into());
        }

        Ok(dataset)
    }
}
