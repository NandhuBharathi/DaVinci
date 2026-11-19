#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Keyword {
    segments: Vec<String>,
}

impl Keyword {
    pub fn parse(input: &str) -> Result<Self, String> {
        let input = input.trim();

        if input.is_empty() {
            return Err("Keyword cannot be empty".into());
        }

        let segments: Vec<String> = input.split("::").map(|s| s.trim().to_string()).collect();

        if segments.iter().any(|s| s.is_empty()) {
            return Err("Invalid empty keyword segment".into());
        }

        if segments[0] != "DAVINCI" {
            return Err("Invalid namespace".into());
        }

        for segment in &segments {
            if !segment
                .chars()
                .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
            {
                return Err(format!("Invalid keyword segment: {}", segment));
            }
        }

        Ok(Self { segments })
    }

    pub fn segments(&self) -> &[String] {
        &self.segments
    }

    pub fn namespace(&self) -> &str {
        &self.segments[0]
    }

    pub fn operation(&self) -> Option<&str> {
        self.segments.last().map(String::as_str)
    }
}
