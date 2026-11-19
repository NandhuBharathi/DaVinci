use super::context::{ContextAnalysis, ContextAnalyzer};

use super::relationship::{Relationship, RelationshipAnalyzer};

use super::sentence_structure::{SentenceAnalyzer, SentenceStructure};

use super::word_meaning::{WordMeaning, WordMeaningResolver};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemanticResult {
    pub original_input: String,

    pub words: Vec<WordMeaning>,

    pub context: ContextAnalysis,

    pub relationships: Vec<Relationship>,

    pub structure: SentenceStructure,
}

pub struct SemanticEngine {
    word_resolver: WordMeaningResolver,

    context_analyzer: ContextAnalyzer,

    relationship_analyzer: RelationshipAnalyzer,

    sentence_analyzer: SentenceAnalyzer,
}

impl SemanticEngine {
    pub fn new() -> Self {
        Self {
            word_resolver: WordMeaningResolver::new(),
            context_analyzer: ContextAnalyzer::new(),
            relationship_analyzer: RelationshipAnalyzer::new(),
            sentence_analyzer: SentenceAnalyzer::new(),
        }
    }

    pub fn understand(&self, input: &str) -> SemanticResult {
        let normalized: Vec<String> = input
            .split_whitespace()
            .map(|word| {
                word.trim_matches(|c: char| !c.is_alphanumeric())
                    .to_lowercase()
            })
            .filter(|word| !word.is_empty())
            .collect();

        let words = self.word_resolver.resolve_all(&normalized);

        let context = self.context_analyzer.analyze(&normalized);

        let relationships = self.relationship_analyzer.analyze(&normalized);

        let structure = self.sentence_analyzer.analyze(input);

        SemanticResult {
            original_input: input.to_string(),
            words,
            context,
            relationships,
            structure,
        }
    }
}

impl Default for SemanticEngine {
    fn default() -> Self {
        Self::new()
    }
}
