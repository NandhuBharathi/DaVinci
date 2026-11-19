#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Knowledge {
    pub key: String,
    pub value: String,
}

impl Knowledge {
    pub fn new(key: String, value: String) -> Result<Self, String> {
        if key.trim().is_empty() || value.trim().is_empty() {
            return Err("Knowledge key/value cannot be empty".into());
        }

        Ok(Self { key, value })
    }
}
