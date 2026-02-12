"""
Entry point for embedding-playground.
Tests both Word2Vec and GloVe download mechanisms.
"""

from src.download import download_word2vec_model, download_glove_model


def main():
    try:
        # Test Word2Vec
        print("=" * 60)
        print("Embedding Playground — Word2Vec Download Test")
        print("=" * 60)

        w2v_path = download_word2vec_model()
        if w2v_path:
            print(f"\nWord2Vec model ready at: {w2v_path}")
        else:
            print("\nWord2Vec model download skipped or failed.")

        # Test GloVe
        print("\n" + "=" * 60)
        print("Embedding Playground — GloVe Download Test")
        print("=" * 60)

        # Test the recommended 100d version
        glove_path = download_glove_model(version="6B.100d")
        if glove_path:
            print(f"\nGloVe model ready at: {glove_path}")
        else:
            print("\nGloVe model download failed.")

    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user (Ctrl+C).")
    except Exception as e:
        print(f"\nUnexpected error: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
