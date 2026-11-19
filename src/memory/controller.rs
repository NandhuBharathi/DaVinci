use super::ltm::{Knowledge, LtmStore};
use super::stm::StmContext;

pub struct MemoryController {
    pub stm: StmContext,
    pub ltm: LtmStore,
}

impl MemoryController {
    pub fn new(stm_capacity: usize, ltm_path: &str) -> Result<Self, String> {
        Ok(Self {
            stm: StmContext::new(stm_capacity)?,
            ltm: LtmStore::open(ltm_path)?,
        })
    }

    pub fn consolidate(&mut self, key: &str) -> Result<(), String> {
        let value = self.stm.recall(key).ok_or("STM entry not found")?;

        let knowledge = Knowledge::new(key.to_string(), value)?;

        self.ltm.insert(knowledge)?;

        self.stm.store.remove(key);

        Ok(())
    }

    pub fn recall_long_term(&self, key: &str) -> Option<String> {
        self.ltm.get(key).map(|item| item.value.clone())
    }
}
