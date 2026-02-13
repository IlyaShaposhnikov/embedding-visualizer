"""
Interactive terminal interface for embedding-playground.
Allows users to query nearest neighbors and word analogies in real time.
"""
from pathlib import Path
import re
from datetime import datetime
from typing import Optional
from gensim.models import KeyedVectors

from src.models import model_info
from src.queries import nearest_neighbors, find_analogies
from src.visualize import visualize_random_words

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

    # Auto-select first available model
    if w2v_model is not None:
        current_model = w2v_model
        model_name = "Word2Vec (GoogleNews)"
    elif glove_model is not None:
        current_model = glove_model
        model_name = "GloVe (6B.100d)"

    if current_model is None:
        print("\nNO MODELS LOADED!")
        print("To use this tool, first download models:")
        print("- Word2Vec:  run download_word2vec_model()")
        print("- GloVe:     run download_glove_model()")
        print("\n   Type 'exit' to quit.\n")
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
                # Switch model: use word2vec / use glove
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
                # Nearest neighbors: nn king 5
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
                # Analogy: ana king man woman 3
                parts = cmd.split()
                if len(parts) < 4:
                    print("Usage: ana <w1> <w2> <w3> [topn]")
                    continue
                w1, w2, w3 = parts[1:4]

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
                except ValueError:
                    print(
                        f"Invalid number: '{parts[4]}'. "
                        "Please use an integer (e.g., 3)."
                    )
                    continue

                if current_model is None:
                    print("No model loaded. Use 'use <model>' first.")
                else:
                    find_analogies(
                        w1, w2, w3, current_model, topn=topn,
                        model_name=model_name
                    )

            elif cmd.startswith("viz "):
                # Usage: viz [n_words] [pca|tsne]
                parts = cmd.split()
                n_words = 50
                method = "pca"
                validation_failed = False

                if len(parts) > 1:
                    try:
                        n_words = int(parts[1])
                        if n_words < 1:
                            print(
                                "Error: n_words must be at least 1 "
                                f"(got {n_words})."
                            )
                            validation_failed = True
                        elif n_words > 200:
                            print(
                                "Warning: n_words capped at 200 "
                                f"(requested {n_words})."
                            )
                            n_words = 200
                    except ValueError:
                        print(
                            f"Error: invalid number '{parts[1]}'. "
                            "Expected integer (e.g., 30)."
                        )
                        validation_failed = True

                if len(parts) > 2 and not validation_failed:
                    method = parts[2].lower()
                    if method not in ("pca", "tsne"):
                        print(
                            f"Error: unknown method '{method}'. "
                            "Use 'pca' or 'tsne'."
                        )
                        validation_failed = True

                if validation_failed:
                    print("Usage: viz [n_words] [pca|tsne]")
                    print("Examples:")
                    print("  viz        → 50 words, PCA")
                    print("  viz 30     → 30 words, PCA")
                    print("  viz 20 tsne → 20 words, t-SNE")
                    continue

                Path(VIZ_SAVE_DIR).mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_model_name = (
                    re.sub(r"[^\w\-]", "_", model_name).strip("_")
                )
                safe_model_name = re.sub(r"_+", "_", safe_model_name)
                filename = (
                    f"viz_{safe_model_name}_{method}_"
                    f"{n_words}words_{timestamp}.png"
                )
                save_path = Path(VIZ_SAVE_DIR) / filename

                if current_model is None:
                    print("No model loaded. Use 'use <model>' first.")
                else:
                    print(
                        f"Generating visualization ({n_words} "
                        f"words, {method.upper()})..."
                    )
                    visualize_random_words(
                        current_model,
                        n_words=n_words,
                        method=method,
                        model_name=model_name,
                        save=save_path,
                    )
                    print(f"Saved to: {save_path.as_posix()}")

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
  use <model>                Switch model: 'word2vec' or 'glove'
  nn <word> [topn]           Nearest neighbors (default topn=5)
  ana <w1> <w2> <w3> [topn]  Word analogy (default topn=3) | w1 - w2 = ? - w3
  viz [n] [pca|tsne]         Visualize random words (default: 50 words, PCA) |
                             → Automatically saved to data/visualizations/
  demo                       Run demonstration queries
  model                      Show current model info
  help                       Show this help
  exit / quit                Exit program
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
    print("DEMONSTRATION: Nearest Neighbors & Analogies")
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
    else:
        print("\nGloVe model not available for demo.")

    print("\n" + "=" * 60)
    print("Demo completed.")
    print("=" * 60)
