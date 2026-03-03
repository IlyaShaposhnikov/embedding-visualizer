"""
Evaluation of embedding models on Google Analogy Test Set.
Computes accuracy for semantic and syntactic analogy questions.
Delegates to services for logic and presentation for formatting.

Source: Mikolov et al., "Efficient Estimation of Word Representations
in Vector Space" (2013)
"""

import logging
import os

from gensim.models import KeyedVectors

from src.services.evaluation import evaluate_model_raw
from src.presentation.formatting import format_evaluation_results

logger = logging.getLogger(__name__)


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

    results = evaluate_model_raw(model, test_file)

    if results is None:
        if model is None:
            print("Load a model first.")
        else:
            print(f"Test file not found: {test_file}")
        return

    formatted_output = format_evaluation_results(results, model_name)
    print(formatted_output)
