#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Intent {
    DefinitionQuestion,
    GeneralQuestion,
    Statement,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuleAnalysis {
    pub tokens: Vec<String>,
    pub intent: Intent,
    pub concept: Option<String>,
    pub rule_trace: Vec<String>,
}

pub struct RuleEngine;

impl RuleEngine {
    pub fn new() -> Self {
        Self
    }

    pub fn analyze(&self, input: &str) -> RuleAnalysis {
        let tokens: Vec<String> = input
            .split_whitespace()
            .map(|word| {
                word.trim_matches(|c: char| !c.is_alphanumeric())
                    .to_lowercase()
            })
            .filter(|word| !word.is_empty())
            .collect();

        let mut trace = Vec::new();

        let has_question_mark = input.trim_end().ends_with('?');

        let is_what_is = tokens.len() >= 3 && tokens[0] == "what" && tokens[1] == "is";

        let intent;

        let concept;

        if is_what_is {
            intent = Intent::DefinitionQuestion;

            concept = Some(tokens[2..].join(" "));

            trace.push("DEFINITION_PATTERN_DETECTED".into());
            trace.push("SUBJECT_IDENTIFIED".into());
        } else if has_question_mark {
            intent = Intent::GeneralQuestion;

            concept = None;

            trace.push("QUESTION_MARK_DETECTED".into());
        } else if !tokens.is_empty() {
            intent = Intent::Statement;

            concept = None;

            trace.push("STATEMENT_PATTERN_DETECTED".into());
        } else {
            intent = Intent::Unknown;

            concept = None;

            trace.push("NO_VALID_CONTENT".into());
        }

        RuleAnalysis {
            tokens,
            intent,
            concept,
            rule_trace: trace,
        }
    }
}

impl Default for RuleEngine {
    fn default() -> Self {
        Self::new()
    }
}
