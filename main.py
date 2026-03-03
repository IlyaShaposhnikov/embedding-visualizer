import sys
import logging

from src.core.logging_config import setup_logging
from src.cli import interactive_shell
from src.download import download_word2vec_model, download_glove_model
from src.models import load_word2vec_model, load_glove_model


def main():
    setup_logging(verbose=False)  # Change to True for debug output
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("Embedding Visualizer — Interactive Shell")
    logger.info("=" * 60)
    logger.info("Models will be downloaded on first use if missing.\n")

    # Preload models (download if missing)
    logger.info("[1] Preparing Word2Vec model...")
    w2v_path = download_word2vec_model()
    w2v_model = (
        load_word2vec_model(w2v_path, use_cached=True) if w2v_path else None
    )

    logger.info("\n[2] Preparing GloVe model (6B.100d)...")
    glove_path = download_glove_model(version="6B.100d")
    glove_model = (
        load_glove_model(glove_path, use_cached=True) if glove_path else None
    )

    # Start interactive shell
    logger.info("\n" + "=" * 60)
    logger.info("Starting interactive shell...")
    logger.info("=" * 60)

    interactive_shell(w2v_model, glove_model)

    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger = logging.getLogger(__name__)
        logger.info("\n\nInterrupted by user (Ctrl+C).")
        sys.exit(130)
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"\nUnexpected error: {type(e).__name__}: {e}")
        sys.exit(1)
