use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WordMeaning {
    pub word: String,
    pub meaning: String,
    pub known: bool,
}

pub struct WordMeaningResolver {
    dictionary: HashMap<String, String>,
}

impl WordMeaningResolver {
    pub fn new() -> Self {
        let mut dictionary = HashMap::new();

        let entries = [
            ("what", "Information request"),
            ("is", "Identity or definition relationship"),
            ("ai", "Artificial Intelligence"),
            ("artificial", "Human-made"),
            ("intelligence", "Ability to learn and reason"),
            ("machine", "Mechanical or computational system"),
            ("learning", "Process of acquiring knowledge"),
            ("data", "Information"),
            ("model", "Learned computational representation"),
            ("engine", "System that performs a function"),
            ("brake", "System used to slow or stop motion"),
            ("disc", "Rotating braking component"),
            ("pad", "Friction material used in braking"),
            ("friction", "Resistance between contacting surfaces"),
            ("kinetic", "Related to motion"),
            ("energy", "Capacity to perform work"),
            ("thermal", "Related to heat"),
            ("converts", "Changes from one form to another"),
            ("into", "Indicates a resulting state"),
            ("uses", "Indicates utilization"),
            ("and", "Connects related items"),
            ("the", "Definite article"),
            ("a", "Indefinite article"),
            ("an", "Indefinite article"),
            ("how", "Method or process question"),
            ("why", "Reason question"),
            ("who", "Person question"),
            ("where", "Location question"),
            ("when", "Time question"),
        ];

        for (word, meaning) in entries {
            dictionary.insert(word.to_string(), meaning.to_string());
        }

        Self { dictionary }
    }

    pub fn resolve(&self, word: &str) -> WordMeaning {
        let normalized = word
            .trim_matches(|c: char| !c.is_alphanumeric())
            .to_lowercase();

        match self.dictionary.get(&normalized) {
            Some(meaning) => WordMeaning {
                word: normalized,
                meaning: meaning.clone(),
                known: true,
            },

            None => WordMeaning {
                word: normalized,
                meaning: "Meaning not found in current dictionary".into(),
                known: false,
            },
        }
    }

    pub fn resolve_all(&self, words: &[String]) -> Vec<WordMeaning> {
        words.iter().map(|word| self.resolve(word)).collect()
    }
}

impl Default for WordMeaningResolver {
    fn default() -> Self {
        Self::new()
    }
}
