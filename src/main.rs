use davinci::understanding::{GuidanceParser, SemanticProcessor, UnderstandingValidator};

fn main() {
    println!("================================");
    println!(" DAVINCI GUIDED UNDERSTANDING ");
    println!("================================");

    let dataset_text = include_str!("../data/training_guidance.dv");

    let dataset = GuidanceParser::parse(dataset_text).expect("Dataset parsing failed");

    let processor = SemanticProcessor::new();

    let validator = UnderstandingValidator::new();

    let input = "brake_disc converts kinetic_energy into thermal_energy through friction";

    let result = processor.analyze(input, &dataset);

    let validation = validator.validate(&result);

    println!("Dataset Samples: {}", dataset.len());

    println!("Input: {}", result.input);

    println!("Word Guidance:");

    for item in &result.guidance {
        println!("{} => {}", item.word, item.meaning_instruction);
    }

    println!("Relationships:");

    for relation in &result.relationships {
        println!(
            "{} -> {} -> {}",
            relation.source, relation.relation, relation.target
        );
    }

    println!("Context: {:?}", result.context);

    println!("Unknown Words: {:?}", result.unknown_words);

    println!("Validation: {}", validation.valid);
}
