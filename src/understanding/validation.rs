use super::semantic::SemanticAnalysis;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationResult {
    pub valid: bool,
    pub errors: Vec<String>,
}

pub struct UnderstandingValidator;

impl UnderstandingValidator {
    pub fn new() -> Self {
        Self
    }

    pub fn validate(&self, result: &SemanticAnalysis) -> ValidationResult {
        let mut errors = Vec::new();

        if result.input.trim().is_empty() {
            errors.push("EMPTY_INPUT".into());
        }

        if result.words.is_empty() {
            errors.push("NO_WORDS_FOUND".into());
        }

        ValidationResult {
            valid: errors.is_empty(),
            errors,
        }
    }
}

impl Default for UnderstandingValidator {
    fn default() -> Self {
        Self::new()
    }
}
