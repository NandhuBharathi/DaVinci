#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SentenceType {
    DefinitionQuestion,
    GeneralQuestion,
    Statement,
    Empty,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SentenceStructure {
    pub sentence_type: SentenceType,
    pub word_count: usize,
    pub has_question_mark: bool,
    pub first_word: Option<String>,
}

pub struct SentenceAnalyzer;

impl SentenceAnalyzer {
    pub fn new() -> Self {
        Self
    }

    pub fn analyze(&self, input: &str) -> SentenceStructure {
        let words: Vec<&str> = input.split_whitespace().collect();

        let first_word = words.first().map(|word| {
            word.trim_matches(|c: char| !c.is_alphanumeric())
                .to_lowercase()
        });

        let has_question_mark = input.trim_end().ends_with('?');

        let definition = words.len() >= 3
            && first_word.as_deref() == Some("what")
            && words[1]
                .trim_matches(|c: char| !c.is_alphanumeric())
                .eq_ignore_ascii_case("is");

        let sentence_type = if words.is_empty() {
            SentenceType::Empty
        } else if definition {
            SentenceType::DefinitionQuestion
        } else if has_question_mark {
            SentenceType::GeneralQuestion
        } else {
            SentenceType::Statement
        };

        SentenceStructure {
            sentence_type,
            word_count: words.len(),
            has_question_mark,
            first_word,
        }
    }
}

impl Default for SentenceAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}
