"""
Functions to format raw data from services into user-readable strings.
Does not perform any business logic or printing.
"""

from typing import List, Tuple, Optional, Dict, Any

from src.services.evaluation import SEMANTIC_SECTIONS, SYNTACTIC_SECTIONS


def format_nearest_neighbors(
    word: str,
    results: List[Tuple[str, float]],
    model_name: Optional[str] = None,
) -> str:
    """
    Format the results of nearest neighbor search for console output.
    """
    model_label = f"{model_name}" if model_name else ""
    output_lines = [f"\n{model_label} | NEAREST NEIGHBORS: '{word}'"]
    output_lines.append("─" * 60)
    for i, (neighbor, sim) in enumerate(results, 1):
        bar = "=" * int(sim * 20)
        output_lines.append(f"{i:2d}. {neighbor:20s} | {sim:.4f} | {bar}")
    output_lines.append("─" * 60)
    return "\n".join(output_lines)


def format_analogy_results(
    w1: str,
    w2: str,
    w3: str,
    results: List[Tuple[str, float]],
    model_name: Optional[str] = None,
) -> str:
    """
    Format the results of analogy solving for console output.
    """
    model_label = f"{model_name}" if model_name else ""
    output_lines = [f"\n{model_label} | ANALOGY: {w1} - {w2} = ? - {w3}"]
    output_lines.append("─" * 60)
    output_lines.append(f"{'#':>2s}  {'Solution':<20s}  {'Similarity':>10s}")
    output_lines.append("─" * 60)
    for i, (candidate, sim) in enumerate(results, 1):
        output_lines.append(f"{i:2d}. {candidate:<20s}  {sim:>10.4f}")
    output_lines.append("─" * 60)
    return "\n".join(output_lines)


def format_evaluation_results(
    results: Dict[str, Any],
    model_name: str
) -> str:
    """
    Format the raw evaluation results
    from evaluate_model_raw into a table string.
    """
    if results is None:
        return ""

    # Unpack results
    section_details = results["section_details"]
    semantic_correct = results["semantic_correct"]
    semantic_total = results["semantic_total"]
    syntactic_correct = results["syntactic_correct"]
    syntactic_total = results["syntactic_total"]
    total_correct = results["total_correct"]
    total_questions = results["total_questions"]
    overall_acc = results["overall_accuracy"]

    # Build output string
    output_lines = []
    output_lines.append("=" * 60)
    output_lines.append(f"EVALUATION: {model_name}")
    output_lines.append(
        "Dataset: Google Analogy Test Set (Mikolov et al., 2013)"
    )
    output_lines.append("=" * 60)

    output_lines.append(
        f"\n{'Section':<35} {'Correct':>8} {'Total':>8} {'Accuracy':>10}"
    )
    output_lines.append("-" * 60)

    for section, corr, tot, acc in section_details:
        section_type = (
            " [SEM]" if section in SEMANTIC_SECTIONS
            else " [SYN]" if section in SYNTACTIC_SECTIONS
            else ""
        )
        label = f"{section}{section_type}"
        output_lines.append(f"{label:<35} {corr:8d} {tot:8d} {acc:9.2f}%")

    output_lines.append("-" * 60)
    if semantic_total > 0:
        sem_acc = results["semantic_accuracy"]
        output_lines.append(
            f"{'SEMANTIC (total)':<35} {semantic_correct:8d} "
            f"{semantic_total:8d} {sem_acc:9.2f}%"
        )
    if syntactic_total > 0:
        syn_acc = results["syntactic_accuracy"]
        output_lines.append(
            f"{'SYNTACTIC (total)':<35} {syntactic_correct:8d} "
            f"{syntactic_total:8d} {syn_acc:9.2f}%"
        )

    if total_questions > 0:
        output_lines.append(
            f"{'OVERALL':<35} {total_correct:8d} "
            f"{total_questions:8d} {overall_acc:9.2f}%"
        )
    else:
        output_lines.append(
            "No questions could be evaluated "
            "(all words missing from vocabulary)"
        )

    output_lines.append("=" * 60)
    output_lines.append("Interpretation:")
    output_lines.append(
        "• Semantic accuracy: measures understanding of meaning relations"
    )
    output_lines.append(
        "• Syntactic accuracy: measures understanding of grammatical patterns"
    )
    output_lines.append("=" * 60)

    return "\n".join(output_lines)
