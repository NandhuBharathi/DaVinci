#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryEntry {
    pub key: String,
    pub value: String,
}

pub struct StmStore {
    entries: Vec<MemoryEntry>,
    capacity: usize,
}

impl StmStore {
    pub fn new(capacity: usize) -> Result<Self, String> {
        if capacity == 0 {
            return Err("STM capacity must be greater than zero".into());
        }

        Ok(Self {
            entries: Vec::new(),
            capacity,
        })
    }

    pub fn insert(&mut self, key: String, value: String) -> Result<(), String> {
        if key.trim().is_empty() || value.trim().is_empty() {
            return Err("STM key/value cannot be empty".into());
        }

        if let Some(entry) = self.entries.iter_mut().find(|e| e.key == key) {
            entry.value = value;
            return Ok(());
        }

        if self.entries.len() >= self.capacity {
            self.entries.remove(0);
        }

        self.entries.push(MemoryEntry { key, value });

        Ok(())
    }

    pub fn get(&self, key: &str) -> Option<&MemoryEntry> {
        self.entries.iter().find(|e| e.key == key)
    }

    pub fn remove(&mut self, key: &str) -> bool {
        if let Some(index) = self.entries.iter().position(|e| e.key == key) {
            self.entries.remove(index);
            return true;
        }

        false
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn entries(&self) -> &[MemoryEntry] {
        &self.entries
    }
}
