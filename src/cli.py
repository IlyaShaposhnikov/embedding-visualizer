"""
Interactive terminal interface for embedding-playground.
Allows users to query nearest neighbors and word analogies in real time.
"""
from datetime import datetime
from pathlib import Path
import re
from typing import Optional

from gensim.models import KeyedVectors

from src.download import download_analogy_test_set
from src.evaluate import evaluate_model
from src.models import model_info
from src.queries import find_analogies, nearest_neighbors
from src.visualize import visualize_word_clusters

MAX_TOPN = 50
VIZ_SAVE_DIR = "data/visualizations"
DEMO_NEIGHBORS = ["king", "france", "computer"]
DEMO_ANALOGIES = [
    ("king", "man", "woman"),
    ("france", "paris", "london"),
    ("moscow", "russia", "tokyo"),
]


def interactive_shell(
    w2v_model: Optional[KeyedVectors],
    glove_model: Optional[KeyedVectors]
) -> None:
    """
    Run interactive command-line interface.
    User can switch between models and execute queries.
    """
    current_model = None
    model_name = "None"

    # Auto-select first available model for immediate usability
    if w2v_model is not None:
        current_model = w2v_model
        model_name = "Word2Vec (GoogleNews)"
    elif glove_model is not None:
        current_model = glove_model
        model_name = "GloVe (6B.100d)"

    if current_model is None:
        print("\nNO MODELS LOADED!")
        print("To use this tool, first download models:")
        print("- Word2Vec: run download_word2vec_model()")
        print("- GloVe: run download_glove_model()")
        print("\nType 'exit' to quit.\n")
    else:
        print(f"\nActive model: {model_name}")
        print(
            "Type 'demo' to run demonstration queries, "
            "'help' for commands, 'exit' to quit.\n"
        )

    while True:
        try:
            cmd = input("\n>>> ").strip().lower()

            if cmd in ("exit", "quit"):
                print("Goodbye!")
                break

            elif cmd == "help":
                _show_help()

            elif cmd == "demo":
                _run_demo(w2v_model, glove_model)

            elif cmd == "model":
                _show_model_status(current_model, model_name)

            elif cmd.startswith("use "):
                # Switch model: use word2vec | use glove
                _, target = cmd.split(maxsplit=1)

                available_models = []
                if w2v_model is not None:
                    available_models.append("word2vec")
                if glove_model is not None:
                    available_models.append("glove")

                if target == "word2vec" and w2v_model is not None:
                    current_model = w2v_model
                    model_name = "Word2Vec (GoogleNews)"
                    print(f"Switched to {model_name}")
                elif target == "glove" and glove_model is not None:
                    current_model = glove_model
                    model_name = "GloVe (6B.100d)"
                    print(f"Switched to {model_name}")
                else:
                    print(f"Model '{target}' not available.")
                    if available_models:
                        print(
                            f"Available models: {', '.join(available_models)}"
                        )
                    else:
                        print("No models loaded. Run download scripts first.")

            elif cmd.startswith("nn "):
                # Nearest neighbors: nn king [5]
                parts = cmd.split()
                if len(parts) < 2:
                    print("Usage: nn <word> [topn]")
                    continue
                word = parts[1]
                try:
                    topn = int(parts[2]) if len(parts) > 2 else 5
                    if topn < 1:
                        print("topn must be at least 1.")
                        continue
                    if topn > MAX_TOPN:
                        print(
                            "Warning: topn capped at "
                            f"{MAX_TOPN} (requested {topn})"
                        )
                        topn = MAX_TOPN
                except ValueError:
                    print(
                        f"Invalid number: '{parts[2]}'. "
                        "Please use an integer (e.g., 5)."
                    )
                    continue
                if current_model is None:
                    print("No model loaded. Use 'use <model>' first.")
                else:
                    nearest_neighbors(
                        word, current_model, topn=topn, model_name=model_name
                    )

            elif cmd.startswith("ana "):
                # Analogy: ana king man woman [topn] [-v] [pca|tsne]
                # Parsing strategy: extract -v flag first,
                # then check for method at the end
                parts = cmd.split()
                if len(parts) < 4:
                    print("Usage: ana <w1> <w2> <w3> [topn] [-v] [pca|tsne]")
                    continue

                # Extract visualization flag before parsing other arguments
                visualize = "-v" in parts
                if visualize:
                    parts.remove("-v")

                # Default method is PCA;
                # check last argument for method override
                method = "pca"
                if parts[-1].lower() in ("pca", "tsne"):
                    method = parts[-1].lower()
                    parts = parts[:-1]

                w1, w2, w3 = parts[1:4]

                # Parse optional topn, default to 3
                try:
                    topn = int(parts[4]) if len(parts) > 4 else 3
                    if topn < 1:
                        print("topn must be at least 1.")
                        continue
                    if topn > MAX_TOPN:
                        print(
                            "Warning: topn capped at "
                            f"{MAX_TOPN} (requested {topn})"
                        )
                        topn = MAX_TOPN
                except (ValueError, IndexError):
                    # Safely handle missing or invalid topn value
                    invalid_val = parts[4] if len(parts) > 4 else "missing"
                    print(
                        f"Invalid number: '{invalid_val}'. "
                        "topn set to default value (3)."
                    )
                    topn = 3

                if current_model is None:
                    print("No model loaded. Use 'use <model>' first.")
                else:
                    # Generate save path if visualization is requested
                    save_path = None
                    if visualize:
                        Path(VIZ_SAVE_DIR).mkdir(parents=True, exist_ok=True)
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        # Sanitize model name for filename
                        safe_model_name = re.sub(
                            r"[^\w\-]", "_", model_name
                        ).strip("_")
                        safe_model_name = re.sub(r"_+", "_", safe_model_name)
                        # Create safe filename from analogy words
                        analogy_str = f"{w1}_{w2}_{w3}"[:40]
                        filename = (
                            f"analogy_{safe_model_name}_{method}_"
                            f"{analogy_str}_top{topn}_{timestamp}.png"
                        )
                        save_path = Path(VIZ_SAVE_DIR) / filename

                    # Execute analogy query with visualization parameters
                    find_analogies(
                        w1, w2, w3, current_model, topn=topn,
                        model_name=model_name,
                        visualize=visualize,
                        method=method,
                        save=save_path
                    )

            elif cmd.startswith("vc "):
                # Usage: vc <word1> [word2 ...] [topn] [pca|tsne]
                # Parse from the end
                parts = cmd.split()
                if len(parts) < 2:
                    print("Usage: vc <word1> [word2 ...] [topn] [pca|tsne]")
                    continue

                words = []
                topn = 3
                method = "pca"

                args = parts[1:]

                # Check last argument for method
                if args[-1].lower() in ("pca", "tsne"):
                    method = args[-1].lower()
                    args = args[:-1]

                # Check new last argument for topn integer
                if args and args[-1].isdigit():
                    topn = int(args[-1])
                    if topn < 1:
                        print("Error: topn must be at least 1.")
                        continue
                    if topn > 20:
                        print(
                            f"Warning: topn capped at 20 (requested {topn})."
                        )
                        topn = 20
                    args = args[:-1]

                # Remaining arguments are seed words
                words = args

                if not words:
                    print("Error: at least one seed word required.")
                    continue

                if current_model is None:
                    print("No model loaded. Use 'use <model>' first.")
                else:
                    # Ensure visualization directory exists
                    Path(VIZ_SAVE_DIR).mkdir(parents=True, exist_ok=True)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    # Sanitize model name for filesystem
                    safe_model_name = re.sub(
                        r"[^\w\-]", "_", model_name
                    ).strip("_")
                    safe_model_name = re.sub(r"_+", "_", safe_model_name)
                    # Create safe filename from seed words (limit length)
                    seed_str = "_".join(words)[:40]
                    filename = (
                        f"clust_{safe_model_name}_{method}_"
                        f"{seed_str}_top{topn}_{timestamp}.png"
                    )
                    save_path = Path(VIZ_SAVE_DIR) / filename

                    print(
                        "Generating cluster visualization "
                        f"for seeds: {', '.join(words)} "
                        f"({topn} neighbors each, {method.upper()})..."
                    )
                    visualize_word_clusters(
                        words,
                        current_model,
                        topn=topn,
                        method=method,
                        model_name=model_name,
                        save=save_path,
                    )
                    print(f"Saved to: {save_path.as_posix()}")

            elif cmd == "eval":
                if current_model is None:
                    print("No model loaded. Use 'use <model>' first.")
                else:
                    test_file = download_analogy_test_set()
                    if test_file:
                        evaluate_model(current_model, test_file, model_name)

            elif cmd == "":
                continue

            else:
                print(f"Unknown command: {cmd}. Type 'help'.")

        except KeyboardInterrupt:
            print("\nUse 'exit' to quit.")
        except ValueError as e:
            print(f"Input error: {e}")
        except (KeyError, IndexError) as e:
            print(f"Model error: {e}")
        except Exception as e:
            print(f"Unexpected error ({type(e).__name__}): {e}")


def _show_help() -> None:
    """Display available commands."""
    help_text = """
COMMANDS:
  use <model>           Switch model: 'word2vec' or 'glove'
  nn <word> [topn]      Nearest neighbors (default topn=5)
  ana <w1> <w2> <w3> [topn] [-v] [m]
                        Word analogy (default topn=3) | w1 - w2 = ? - w3
                        • -v   : visualize results in 2D space (pca method)
                        • [m]  : method 'pca' or 'tsne' (default pca)
                        • Automatically saved to data/visualizations/
  vc <w1> [w2 ...] [n] [m]
                        Visualize semantic clusters:
                        • <w1> : seed words (min 1)
                        • [n]  : neighbors per seed (default 3, max 20)
                        • [m]  : method 'pca' or 'tsne' (default pca)
                        • Automatically saved to data/visualizations/
  demo                  Run full demonstration
                        • nearest neighbors, solve analogies, semantic clusters
  model                 Show current model info
  eval                  Evaluate current model on Google Analogy Test Set
  help                  Show this help
  exit / quit           Exit program
"""
    print(help_text)


def _show_model_status(
    model: Optional[KeyedVectors],
    name: str
) -> None:
    """Display current model status."""
    if model is None:
        print("No model loaded.")
    else:
        model_info(model, name)


def _run_demo(
        w2v_model: Optional[KeyedVectors],
        glove_model: Optional[KeyedVectors]
) -> None:
    """Run demonstration queries for available models."""
    print("\n" + "=" * 60)
    print("DEMONSTRATION: Nearest Neighbors, Analogies & Clusters")
    print("=" * 60)

    # Word2Vec demo
    if w2v_model is not None:
        print("\n[Word2Vec Demo]")
        print("-" * 60)
        model_info(w2v_model, "Word2Vec (GoogleNews)")

        for word in DEMO_NEIGHBORS:
            nearest_neighbors(
                word, w2v_model, topn=5, model_name="Word2Vec (GoogleNews)"
            )

        print("\nWord2Vec Analogies")
        for w1, w2, w3 in DEMO_ANALOGIES:
            find_analogies(
                w1, w2, w3, w2v_model, topn=3,
                model_name="Word2Vec (GoogleNews)"
            )

        print("\nWord2Vec Cluster Visualization")
        print(f"Seeds: {', '.join(DEMO_NEIGHBORS)}")
        visualize_word_clusters(
            DEMO_NEIGHBORS,
            w2v_model,
            topn=3,
            method="pca",
            model_name="Word2Vec (GoogleNews)",
            save=None,
        )
    else:
        print("\nWord2Vec model not available for demo.")

    # GloVe demo
    if glove_model is not None:
        print("\n[GloVe Demo]")
        print("-" * 60)
        model_info(glove_model, "GloVe (6B.100d)")

        for word in DEMO_NEIGHBORS:
            nearest_neighbors(
                word, glove_model, topn=5, model_name="GloVe (6B.100d)"
            )

        print("\nGloVe Analogies")
        for w1, w2, w3 in DEMO_ANALOGIES:
            find_analogies(
                w1, w2, w3, glove_model, topn=3, model_name="GloVe (6B.100d)"
            )

        print("\nGloVe Cluster Visualization")
        print(f"Seeds: {', '.join(DEMO_NEIGHBORS)}")
        visualize_word_clusters(
            DEMO_NEIGHBORS,
            glove_model,
            topn=3,
            method="pca",
            model_name="GloVe (6B.100d)",
            save=None,
        )
    else:
        print("\nGloVe model not available for demo.")

    print("\n" + "=" * 60)
    print("Demo completed.")
    print("=" * 60)
