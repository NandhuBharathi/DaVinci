use davinci::memory::{Knowledge, LtmStore, MemoryController, StmStore};

fn temp_path(name: &str) -> String {
    format!("/tmp/davinci_memory_{}_{}.db", name, std::process::id())
}

#[test]
fn stm_capacity_is_enforced() {
    let mut stm = StmStore::new(2).unwrap();

    stm.insert("a".into(), "one".into()).unwrap();
    stm.insert("b".into(), "two".into()).unwrap();
    stm.insert("c".into(), "three".into()).unwrap();

    assert_eq!(stm.len(), 2);
    assert!(stm.get("a").is_none());
    assert!(stm.get("c").is_some());
}

#[test]
fn stm_updates_existing_entry() {
    let mut stm = StmStore::new(2).unwrap();

    stm.insert("a".into(), "old".into()).unwrap();
    stm.insert("a".into(), "new".into()).unwrap();

    assert_eq!(stm.len(), 1);
    assert_eq!(stm.get("a").unwrap().value, "new");
}

#[test]
fn stm_rejects_empty_values() {
    let mut stm = StmStore::new(2).unwrap();

    assert!(stm.insert("".into(), "value".into()).is_err());
    assert!(stm.insert("key".into(), "".into()).is_err());
}

#[test]
fn stm_remove_and_clear() {
    let mut stm = StmStore::new(3).unwrap();

    stm.insert("a".into(), "one".into()).unwrap();
    assert!(stm.remove("a"));

    stm.insert("b".into(), "two".into()).unwrap();
    stm.clear();

    assert!(stm.is_empty());
}

#[test]
fn ltm_persistence_survives_reopen() {
    let path = temp_path("persist");
    let _ = std::fs::remove_file(&path);

    {
        let mut ltm = LtmStore::open(&path).unwrap();

        ltm.insert(Knowledge::new("brake_disc".into(), "braking_component".into()).unwrap())
            .unwrap();
    }

    let ltm = LtmStore::open(&path).unwrap();

    assert_eq!(ltm.get("brake_disc").unwrap().value, "braking_component");

    let _ = std::fs::remove_file(path);
}

#[test]
fn ltm_updates_duplicate_key() {
    let path = temp_path("update");
    let _ = std::fs::remove_file(&path);

    let mut ltm = LtmStore::open(&path).unwrap();

    ltm.insert(Knowledge::new("x".into(), "old".into()).unwrap())
        .unwrap();
    ltm.insert(Knowledge::new("x".into(), "new".into()).unwrap())
        .unwrap();

    assert_eq!(ltm.len(), 1);
    assert_eq!(ltm.get("x").unwrap().value, "new");

    let _ = std::fs::remove_file(path);
}

#[test]
fn ltm_search_finds_matching_knowledge() {
    let path = temp_path("search");
    let _ = std::fs::remove_file(&path);

    let mut ltm = LtmStore::open(&path).unwrap();

    ltm.insert(Knowledge::new("brake_disc".into(), "friction_component".into()).unwrap())
        .unwrap();

    assert_eq!(ltm.search("friction").len(), 1);

    let _ = std::fs::remove_file(path);
}

#[test]
fn controller_consolidates_stm_into_ltm() {
    let path = temp_path("controller");
    let _ = std::fs::remove_file(&path);

    let mut controller = MemoryController::new(4, &path).unwrap();

    controller.stm.remember("concept", "learned_value").unwrap();

    controller.consolidate("concept").unwrap();

    assert!(controller.stm.recall("concept").is_none());

    assert_eq!(
        controller.recall_long_term("concept").unwrap(),
        "learned_value"
    );

    let _ = std::fs::remove_file(path);
}

#[test]
fn controller_rejects_missing_stm_entry() {
    let path = temp_path("missing");
    let _ = std::fs::remove_file(&path);

    let mut controller = MemoryController::new(4, &path).unwrap();

    assert!(controller.consolidate("missing").is_err());

    let _ = std::fs::remove_file(path);
}

#[test]
fn invalid_stm_capacity_is_rejected() {
    assert!(StmStore::new(0).is_err());
}
