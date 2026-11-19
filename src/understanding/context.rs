#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContextType {
    Definition,
    Technical,
    GeneralQuestion,
    Statement,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContextAnalysis {
    pub context_type: ContextType,
    pub subject: Option<String>,
    pub evidence: Vec<String>,
}

pub struct ContextAnalyzer;

impl ContextAnalyzer {
    pub fn new() -> Self {
        Self
    }

    pub fn analyze(&self, words: &[String]) -> ContextAnalysis {
        let mut evidence = Vec::new();

        let definition = words.len() >= 3 && words[0] == "what" && words[1] == "is";

        let technical_terms = [
            "ai", "machine", "engine", "data", "model", "neural", "brake", "energy", "friction",
        ];

        let technical = words
            .iter()
            .any(|word| technical_terms.contains(&word.as_str()));

        if definition {
            evidence.push("WHAT_IS_PATTERN".into());

            return ContextAnalysis {
                context_type: ContextType::Definition,
                subject: Some(words[2..].join(" ")),
                evidence,
            };
        }

        if technical {
            evidence.push("TECHNICAL_TERM_DETECTED".into());

            return ContextAnalysis {
                context_type: ContextType::Technical,
                subject: None,
                evidence,
            };
        }

        if words
            .iter()
            .any(|w| ["how", "why", "who", "where", "when"].contains(&w.as_str()))
        {
            evidence.push("QUESTION_WORD_DETECTED".into());

            return ContextAnalysis {
                context_type: ContextType::GeneralQuestion,
                subject: None,
                evidence,
            };
        }

        if !words.is_empty() {
            evidence.push("CONTENT_PRESENT".into());

            return ContextAnalysis {
                context_type: ContextType::Statement,
                subject: None,
                evidence,
            };
        }

        ContextAnalysis {
            context_type: ContextType::Unknown,
            subject: None,
            evidence: vec!["EMPTY_CONTENT".into()],
        }
    }
}

impl Default for ContextAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}
