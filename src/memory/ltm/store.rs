use super::knowledge::Knowledge;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::Path;

pub struct LtmStore {
    knowledge: Vec<Knowledge>,
    path: String,
}

impl LtmStore {
    pub fn open(path: &str) -> Result<Self, String> {
        let file_path = Path::new(path);

        if let Some(parent) = file_path.parent() {
            fs::create_dir_all(parent).map_err(|e| format!("Directory error: {e}"))?;
        }

        let mut knowledge = Vec::new();

        if file_path.exists() {
            let content =
                fs::read_to_string(file_path).map_err(|e| format!("LTM read error: {e}"))?;

            for line in content.lines() {
                let (key, value) = line.split_once('\t').ok_or("Invalid LTM record")?;

                knowledge.push(Knowledge::new(key.to_string(), value.to_string())?);
            }
        }

        Ok(Self {
            knowledge,
            path: path.to_string(),
        })
    }

    pub fn insert(&mut self, item: Knowledge) -> Result<(), String> {
        if let Some(existing) = self
            .knowledge
            .iter_mut()
            .find(|entry| entry.key == item.key)
        {
            existing.value = item.value;
        } else {
            self.knowledge.push(item);
        }

        self.persist()
    }

    pub fn get(&self, key: &str) -> Option<&Knowledge> {
        self.knowledge.iter().find(|entry| entry.key == key)
    }

    pub fn search(&self, query: &str) -> Vec<Knowledge> {
        let query = query.to_lowercase();

        self.knowledge
            .iter()
            .filter(|entry| {
                entry.key.to_lowercase().contains(&query)
                    || entry.value.to_lowercase().contains(&query)
            })
            .cloned()
            .collect()
    }

    pub fn len(&self) -> usize {
        self.knowledge.len()
    }

    pub fn is_empty(&self) -> bool {
        self.knowledge.is_empty()
    }

    pub fn persist(&self) -> Result<(), String> {
        let temp_path = format!("{}.tmp", self.path);

        let mut file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(&temp_path)
            .map_err(|e| format!("LTM write error: {e}"))?;

        for item in &self.knowledge {
            if item.key.contains('\t')
                || item.value.contains('\t')
                || item.key.contains('\n')
                || item.value.contains('\n')
            {
                return Err("LTM values cannot contain tabs or newlines".into());
            }

            writeln!(file, "{}\t{}", item.key, item.value)
                .map_err(|e| format!("LTM write error: {e}"))?;
        }

        file.sync_all()
            .map_err(|e| format!("LTM sync error: {e}"))?;

        fs::rename(&temp_path, &self.path).map_err(|e| format!("LTM replace error: {e}"))?;

        Ok(())
    }
}
