use super::guidance::{GuidanceDataset, RelationGuidance, WordGuidance};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemanticAnalysis {
    pub input: String,
    pub words: Vec<String>,
    pub guidance: Vec<WordGuidance>,
    pub relationships: Vec<RelationGuidance>,
    pub context: Vec<String>,
    pub unknown_words: Vec<String>,
}

pub struct SemanticProcessor;

impl SemanticProcessor {
    pub fn new() -> Self {
        Self
    }

    pub fn analyze(&self, input: &str, dataset: &GuidanceDataset) -> SemanticAnalysis {
        let words: Vec<String> = input
            .split_whitespace()
            .map(|word| {
                word.trim_matches(|c: char| !c.is_alphanumeric() && c != '_')
                    .to_lowercase()
            })
            .filter(|word| !word.is_empty())
            .collect();

        let mut guidance = Vec::new();
        let mut unknown_words = Vec::new();
        let mut relationships = Vec::new();
        let mut contexts = Vec::new();

        for word in &words {
            let found = dataset.find_word_guidance(word);

            if found.is_empty() {
                unknown_words.push(word.clone());
            } else {
                guidance.extend(found);
            }
        }

        for example in &dataset.examples {
            let matches_content = example.content.split_whitespace().any(|item| {
                words.iter().any(|word| {
                    item.trim_matches(|c: char| !c.is_alphanumeric() && c != '_')
                        .eq_ignore_ascii_case(word)
                })
            });

            if matches_content {
                relationships.extend(example.relationships.clone());

                if !example.context_instruction.is_empty() {
                    contexts.push(example.context_instruction.clone());
                }
            }
        }

        SemanticAnalysis {
            input: input.to_string(),
            words,
            guidance,
            relationships,
            context: contexts,
            unknown_words,
        }
    }
}

impl Default for SemanticProcessor {
    fn default() -> Self {
        Self::new()
    }
}
