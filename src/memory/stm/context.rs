use super::store::StmStore;

pub struct StmContext {
    pub store: StmStore,
}

impl StmContext {
    pub fn new(capacity: usize) -> Result<Self, String> {
        Ok(Self {
            store: StmStore::new(capacity)?,
        })
    }

    pub fn remember(&mut self, key: &str, value: &str) -> Result<(), String> {
        self.store.insert(key.to_string(), value.to_string())
    }

    pub fn recall(&self, key: &str) -> Option<String> {
        self.store.get(key).map(|entry| entry.value.clone())
    }
}
