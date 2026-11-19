#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WordGuidance {
    pub word: String,
    pub meaning_instruction: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelationGuidance {
    pub source: String,
    pub relation: String,
    pub target: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrainingExample {
    pub id: String,
    pub content: String,
    pub word_guidance: Vec<WordGuidance>,
    pub relationships: Vec<RelationGuidance>,
    pub context_instruction: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GuidanceDataset {
    pub examples: Vec<TrainingExample>,
}

impl GuidanceDataset {
    pub fn new() -> Self {
        Self {
            examples: Vec::new(),
        }
    }

    pub fn add(&mut self, example: TrainingExample) {
        self.examples.push(example);
    }

    pub fn find_word_guidance(&self, word: &str) -> Vec<WordGuidance> {
        self.examples
            .iter()
            .flat_map(|example| example.word_guidance.iter())
            .filter(|item| item.word.eq_ignore_ascii_case(word))
            .cloned()
            .collect()
    }

    pub fn find_example(&self, id: &str) -> Option<&TrainingExample> {
        self.examples.iter().find(|example| example.id == id)
    }

    pub fn len(&self) -> usize {
        self.examples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.examples.is_empty()
    }
}

impl Default for GuidanceDataset {
    fn default() -> Self {
        Self::new()
    }
}
