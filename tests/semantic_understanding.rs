use davinci::understanding::*;

const SAMPLE: &str = "
SAMPLE
ID: TEST_001
CONTENT: A brake disc converts kinetic energy into thermal energy.
WORD_GUIDANCE:
brake_disc\tbraking_component
converts\tenergy_transformation
kinetic_energy\tmotion_energy
thermal_energy\theat_energy
RELATIONSHIPS:
brake_disc\tinteracts_through\tfriction
kinetic_energy\ttransforms_into\tthermal_energy
CONTEXT: technical_explanation
END_SAMPLE
";

#[test]
fn parse_training_dataset() {
    let dataset = GuidanceParser::parse(SAMPLE).unwrap();

    assert_eq!(dataset.len(), 1);
}

#[test]
fn load_word_guidance() {
    let dataset = GuidanceParser::parse(SAMPLE).unwrap();

    let result = dataset.find_word_guidance("brake_disc");

    assert_eq!(result.len(), 1);

    assert_eq!(result[0].meaning_instruction, "braking_component");
}

#[test]
fn load_relationship_guidance() {
    let dataset = GuidanceParser::parse(SAMPLE).unwrap();

    let example = dataset.find_example("TEST_001").unwrap();

    assert_eq!(example.relationships.len(), 2);
}

#[test]
fn reject_invalid_dataset() {
    let result = GuidanceParser::parse("SAMPLE\nID: INVALID\nEND_SAMPLE");

    assert!(result.is_err());
}

#[test]
fn reject_incomplete_dataset() {
    let result = GuidanceParser::parse("SAMPLE\nID: TEST\nCONTENT: Some content");

    assert!(result.is_err());
}

#[test]
fn semantic_guidance_processing() {
    let dataset = GuidanceParser::parse(SAMPLE).unwrap();

    let processor = SemanticProcessor::new();

    let result = processor.analyze(
        "brake_disc converts kinetic_energy into thermal_energy",
        &dataset,
    );

    assert!(!result.guidance.is_empty());

    assert!(!result.relationships.is_empty());
}

#[test]
fn unknown_words_are_identified() {
    let dataset = GuidanceParser::parse(SAMPLE).unwrap();

    let processor = SemanticProcessor::new();

    let result = processor.analyze("unknown_concept", &dataset);

    assert_eq!(result.unknown_words, vec!["unknown_concept".to_string()]);
}

#[test]
fn validate_semantic_input() {
    let dataset = GuidanceParser::parse(SAMPLE).unwrap();

    let processor = SemanticProcessor::new();

    let validator = UnderstandingValidator::new();

    let result = processor.analyze("brake_disc converts thermal_energy", &dataset);

    assert!(validator.validate(&result).valid);
}

#[test]
fn empty_input_is_rejected() {
    let dataset = GuidanceParser::parse(SAMPLE).unwrap();

    let processor = SemanticProcessor::new();

    let validator = UnderstandingValidator::new();

    let result = processor.analyze("", &dataset);

    assert!(!validator.validate(&result).valid);
}
