"""
Pure business logic for model evaluation on the Google Analogy Test Set.
Does not perform any formatting or printing.
"""

import logging
import os
from typing import Dict, List, Tuple, Optional

from gensim.models import KeyedVectors

from src.core.config import settings
from src.data.data_extraction import get_analogy_solution

logger = logging.getLogger(__name__)


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
        for _, line in enumerate(f, 1):
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
                # Could log this or collect warnings, but for now just ignore
                pass

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


def evaluate_model_raw(
    model: KeyedVectors,
    test_file: str,
) -> Optional[Dict[str, any]]:
    """
    Evaluate embedding model on Google Analogy Test Set.
    Returns structured results dictionary instead of printing.
    """
    if model is None:
        # Return None or raise an exception depending on how caller handles it
        # For now, returning None seems appropriate
        return None

    if not os.path.exists(test_file):
        # Return None or raise an exception
        return None

    try:
        sections = parse_questions_file(test_file)
    except Exception:
        # Could log the error, but for now just return None
        return None

    section_results = []
    semantic_correct = semantic_total = 0
    syntactic_correct = syntactic_total = 0

    for section_name, questions in sections.items():
        correct, total = evaluate_section(questions, model)

        if total > 0:
            acc = correct / total * 100
            section_results.append((section_name, correct, total, acc))

            # Classify section based on its name
            if section_name in settings.AnalogyTestSet.SEMANTIC_SECTIONS:
                semantic_correct += correct
                semantic_total += total
            elif section_name in settings.AnalogyTestSet.SYNTACTIC_SECTIONS:
                syntactic_correct += correct
                syntactic_total += total
            # Sections not listed are ignored in aggregate stats

    total_correct = semantic_correct + syntactic_correct
    total_questions = semantic_total + syntactic_total
    overall_acc = (
        (total_correct / total_questions * 100)
        if total_questions > 0 else 0
    )

    # Prepare structured results
    results = {
        "section_details": section_results,
        "semantic_correct": semantic_correct,
        "semantic_total": semantic_total,
        "semantic_accuracy": (
            (semantic_correct / semantic_total * 100)
            if semantic_total > 0 else 0
        ),
        "syntactic_correct": syntactic_correct,
        "syntactic_total": syntactic_total,
        "syntactic_accuracy": (
            (syntactic_correct / syntactic_total * 100)
            if syntactic_total > 0 else 0
        ),
        "total_correct": total_correct,
        "total_questions": total_questions,
        "overall_accuracy": overall_acc,
        "all_sections_count": len(sections),
        "processed_sections_count": len(section_results),
    }

    return results
