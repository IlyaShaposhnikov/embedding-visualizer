"""
Evaluation of embedding models on Google Analogy Test Set.
Computes accuracy for semantic and syntactic analogy questions.
Source: Mikolov et al., "Efficient Estimation of Word Representations
in Vector Space" (2013)
"""

import logging
import os
from typing import Dict, List, Tuple

from gensim.models import KeyedVectors

from src.data.data_extraction import get_analogy_solution

logger = logging.getLogger(__name__)

# Classification of sections based on the categories
# defined in the original paper.
# These sets are used to separate semantic and syntactic accuracy.
SEMANTIC_SECTIONS = {
    'capital-common-countries', 'capital-world', 'currency',
    'city-in-state', 'family', 'gram6-nationality-adjective'
}
SYNTACTIC_SECTIONS = {
    'gram1-adjective-to-adverb', 'gram2-opposite', 'gram3-comparative',
    'gram4-superlative', 'gram5-present-participle', 'gram7-past-tense',
    'gram8-plural', 'gram9-plural-verbs'
}


def parse_questions_file(
        file_path: str
) -> Dict[str, List[Tuple[str, str, str, str]]]:
    """
    Parse Google Analogy test file into sections.

    File format:
      - Lines starting with ':' indicate a section name
      (e.g., ":capital-common-countries").
      - Each subsequent line contains four tokens: w1 w2 w3 expected_answer.
      - Empty lines and lines starting with '//' are ignored.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Test file not found: {file_path}")

    sections = {}
    current_section = None
    questions = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()

            # Skip empty lines and comment lines (//)
            if not line or line.startswith('//'):
                continue

            # Section header
            if line.startswith(':'):
                if current_section and questions:
                    sections[current_section] = questions

                current_section = line[1:].strip()
                questions = []
                continue

            # Normal question line: w1 w2 w3 expected
            parts = line.split()
            if len(parts) == 4:
                w1, w2, w3, expected = parts
                questions.append((w1, w2, w3, expected))
            elif parts:
                # If line has tokens but not exactly 4, it's malformed
                logger.warning(
                    f"Warning: line {line_num} has unexpected format: '{line}'"
                )

    # Save the last section
    if current_section and questions:
        sections[current_section] = questions

    if not sections:
        raise ValueError(f"No valid questions found in {file_path}")

    total_questions = sum(len(q) for q in sections.values())
    logger.info(
        f"Parsed {len(sections)} sections with {total_questions:,} questions"
    )
    return sections


def evaluate_section(
    questions: List[Tuple[str, str, str, str]],
    model: KeyedVectors,
    verbose: bool = False
) -> Tuple[int, int]:
    """
    Evaluate model on analogy questions for a single section.
    Questions where any of the four words
    is missing from the vocabulary are skipped.
    """
    correct = 0
    total = 0

    for w1, w2, w3, expected in questions:
        # Use topn=1 because we only care about the top prediction
        results = get_analogy_solution(w1, w2, w3, model, topn=1)

        # If results is None, at least one word was missing -> skip
        if results is None:
            continue

        predicted, _ = results[0]

        if predicted.lower() == expected.lower():
            correct += 1

        total += 1

    return correct, total


def evaluate_model(
    model: KeyedVectors,
    test_file: str,
    model_name: str = "Model"
) -> None:
    """Evaluate embedding model on Google Analogy Test Set."""
    if model is None:
        print("Load a model first.")
        return

    if not os.path.exists(test_file):
        print(f"Test file not found: {test_file}")
        return

    print("=" * 60)
    print(f"EVALUATION: {model_name}")
    print("Dataset: Google Analogy Test Set (Mikolov et al., 2013)")
    print("=" * 60)

    try:
        sections = parse_questions_file(test_file)
    except Exception as e:
        print(f"Error parsing test file: {e}")
        return

    section_results = []
    semantic_correct = semantic_total = 0
    syntactic_correct = syntactic_total = 0

    for section_name, questions in sections.items():
        correct, total = evaluate_section(questions, model)

        if total > 0:
            acc = correct / total * 100
            section_results.append((section_name, correct, total, acc))

            # Classify section based on its name
            if section_name in SEMANTIC_SECTIONS:
                semantic_correct += correct
                semantic_total += total
            elif section_name in SYNTACTIC_SECTIONS:
                syntactic_correct += correct
                syntactic_total += total
            # Sections not listed are ignored in aggregate stats

    print(
        f"\n{'Section':<35} {'Correct':>8} {'Total':>8} {'Accuracy':>10}"
    )
    print("-" * 60)

    for section, corr, tot, acc in section_results:
        section_type = (
            " [SEM]" if section in SEMANTIC_SECTIONS
            else " [SYN]" if section in SYNTACTIC_SECTIONS
            else ""
        )
        label = f"{section}{section_type}"
        print(f"{label:<35} {corr:8d} {tot:8d} {acc:9.2f}%")

    print("-" * 60)
    if semantic_total > 0:
        sem_acc = semantic_correct / semantic_total * 100
        print(
            f"{'SEMANTIC (total)':<35} {semantic_correct:8d} "
            f"{semantic_total:8d} {sem_acc:9.2f}%"
        )
    if syntactic_total > 0:
        syn_acc = syntactic_correct / syntactic_total * 100
        print(
            f"{'SYNTACTIC (total)':<35} {syntactic_correct:8d} "
            f"{syntactic_total:8d} {syn_acc:9.2f}%"
        )

    total_correct = semantic_correct + syntactic_correct
    total_questions = semantic_total + syntactic_total
    if total_questions > 0:
        overall_acc = total_correct / total_questions * 100
        print(
            f"{'OVERALL':<35} {total_correct:8d} "
            f"{total_questions:8d} {overall_acc:9.2f}%"
        )
    else:
        print(
            "No questions could be evaluated "
            "(all words missing from vocabulary)"
        )

    print("=" * 60)
    print("Interpretation:")
    print(
        "• Semantic accuracy: measures understanding of meaning relations"
    )
    print(
        "• Syntactic accuracy: measures understanding of grammatical patterns"
    )
    print("=" * 60)
