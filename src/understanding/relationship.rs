#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Relationship {
    pub source: String,
    pub relation: String,
    pub target: String,
}

pub struct RelationshipAnalyzer;

impl RelationshipAnalyzer {
    pub fn new() -> Self {
        Self
    }

    pub fn analyze(&self, words: &[String]) -> Vec<Relationship> {
        let mut relationships = Vec::new();

        if words.len() >= 3 {
            if words[0] == "what" && words[1] == "is" {
                relationships.push(Relationship {
                    source: words[2..].join(" "),
                    relation: "REQUESTS_DEFINITION".into(),
                    target: "DEFINITION".into(),
                });
            }
        }

        for i in 0..words.len() {
            if i + 2 < words.len() && words[i + 1] == "is" {
                relationships.push(Relationship {
                    source: words[i].clone(),
                    relation: "IS_A".into(),
                    target: words[i + 2].clone(),
                });
            }

            if i + 2 < words.len() && words[i + 1] == "uses" {
                relationships.push(Relationship {
                    source: words[i].clone(),
                    relation: "USES".into(),
                    target: words[i + 2].clone(),
                });
            }
        }

        for i in 0..words.len() {
            if i + 3 < words.len() && words[i + 1] == "converts" && words[i + 2] == "into" {
                relationships.push(Relationship {
                    source: words[i].clone(),
                    relation: "CONVERTS_INTO".into(),
                    target: words[i + 3].clone(),
                });
            }
        }

        relationships
    }
}

impl Default for RelationshipAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}
