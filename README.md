**[Russian Version / На русском](README.ru.md)**

# Embedding Playground

**Interactive toolkit for exploring word embeddings: Word2Vec, GloVe, nearest neighbors, analogies, and 2D visualizations.**


## Features

- **Automatic Download** of pre-trained models with integrity checks:
  - Word2Vec GoogleNews (3M words, 300d)
  - GloVe 6B (400K words, 50/100/200/300d variants)
- **Smart Caching** – models saved in Gensim binary format for instant subsequent loads
- **Nearest Neighbors** – find semantically similar words with visual similarity bars
- **Analogies** – solve `king - man + woman = ?` with 2D vector visualization (`-v` flag)
- **Semantic Clusters** – plot seed words + neighbors using PCA/t-SNE
- **Evaluation** – test on Google Analogy Test Set (19,544 questions) with semantic/syntactic breakdown
- **Interactive CLI** – intuitive shell with contextual help and demo mode


## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/IlyaShaposhnikov/embedding-playground.git
cd embedding-playground
python -m venv .venv
source venv/Scripts/activate
pip install -r requirements.txt
```

### 2. Run the interactive shell

```bash
python main.py
```

On first run, models download automatically (~1.5 GB for Word2Vec, ~800 MB for GloVe). Subsequent runs use cached binaries.

### 3. Try commands

```bash
>>> nn king 5                     # 5 nearest neighbors of 'king'
>>> ana paris france berlin -v    # paris - france = ? - berlin + visualization
>>> vc king queen computer 3 pca  # clusters for 3 seeds with 3 neighbors each (PCA)
>>> eval                          # evaluate on Google Analogy Test Set
>>> demo                          # full demonstration (neighbors, analogies, clusters)
```


## Command Reference

| Command | Description |
|---------|-------------|
| `use <model>` | Switch active model: `word2vec` or `glove` |
| `nn <word> [topn]` | Nearest neighbors (default `topn=5`) |
| `ana <w1> <w2> <w3> [topn] [-v]` | Solve analogy `w1 - w2 = ? - w3`<br>• `-v`: visualize vector relationships (PCA) |
| `vc <w1> [w2 ...] [n] [m]` | Visualize semantic clusters:<br>• Seeds: `w1`... (min 1)<br>• `[n]`: neighbors per seed (default 3, max 20)<br>• `[m]`: method `pca` or `tsne` (default `pca`)<br>- **PCA** — _Linear projection_ that preserves **global structure** by mapping vectors onto axes of maximum variance. Fast, but may blur local clusters.<br>- **t-SNE** — _Non-linear projection_ that preserves **local neighborhoods** by modeling pairwise similarities. Slower, but reveals fine-grained clusters at the cost of global geometry.<br>→ **Auto-saved** to `data/visualizations/` |
| `eval` | Evaluate current model on Google Analogy Test Set |
| `model` | Show model info (vocab size, dimension, memory usage) |
| `demo` | Run full demonstration (nearest neighbors, analogies, clusters for both models) |
| `help` | Show command reference |
| `exit` / `quit` | Exit program |


## Example Outputs

### Nearest neighbors with similarity bars
```
Word2Vec (GoogleNews) | NEAREST NEIGHBORS: 'king'
────────────────────────────────────────────────────────────
 1. queen                | 0.7660 | ================
 2. prince               | 0.7421 | ===============
 3. kings                | 0.7285 | ===============
 4. monarch              | 0.7123 | ==============
 5. crown                | 0.6987 | ==============
────────────────────────────────────────────────────────────
```

### Analogy solution
```
Word2Vec (GoogleNews) | ANALOGY: king - man = ? - woman
────────────────────────────────────────────────────────────
#   Solution                 Similarity
────────────────────────────────────────────────────────────
 1. queen                       0.7660
 2. monarch                     0.7421
 3. prince                      0.7285
────────────────────────────────────────────────────────────
```


## Project Structure
```
embedding-playground/
├── data/                    # Models (downloaded by user) & visualizations
│   ├── GoogleNews-vectors-negative300.bin      # Word2Vec binary (3.4 GB)
│   ├── glove.6B.100d.txt                       # GloVe vectors (331 MB)
│   └── visualizations/                         # Auto-saved plots
├── src/
│   ├── cli.py              # Interactive shell & command parsing
│   ├── download.py         # Model download with size verification & mirrors
│   ├── evaluate.py         # Google Analogy Test Set evaluation
│   ├── models.py           # Model loading with caching
│   ├── queries.py          # Core operations (nearest neighbors, analogies)
│   └── visualize.py        # PCA/t-SNE projection & cluster/analogy plots
├── main.py                 # Entry point
├── requirements.txt        # Dependencies
├── README.md               # Project documentation (in English)
└── README.ru.md            # Project documentation (in Russian)
```

## Troubleshooting

### Google Drive quota exceeded
Word2Vec download may fail with:
```
Too many users have viewed or downloaded this file recently...
```
**Solution:**
Use [manual download](https://github.com/mmihaltz/word2vec-GoogleNews-vectors/raw/master/GoogleNews-vectors-negative300.bin.gz)

### Low accuracy on GloVe 6B.100d
GloVe 6B.100d has limited vocabulary (~400K words) and 100d dimensionality:
- **Geographic names often missing** → some semantic sections skipped during `eval`
- **Expected accuracy**: ~2-5% overall (vs 65-75% for Word2Vec GoogleNews 300d)
- **Recommendation**: Use Word2Vec for serious evaluation; GloVe 100d is suitable for demonstrations

### Memory requirements
| Model | RAM required | Load time (first) | Load time (cached) |
|-------|--------------|-------------------|--------------------|
| Word2Vec GoogleNews | ~4.2 GB | 3-5 min | ~10 sec |
| GloVe 6B.100d | ~1.2 GB | 1-2 min | ~5 sec |

Ensure ≥6 GB free RAM for comfortable usage with both models.


## References & Citations

### Core Papers
- Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781). *ICLR*.
- Pennington, J., Socher, R., & Manning, C. D. (2014). [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf). *EMNLP*.

### Datasets
- **Word2Vec GoogleNews**: Trained on Google News corpus (3B words), 3M vocabulary, 300d vectors
- **GloVe 6B**: Trained on Wikipedia 2014 + Gigaword 5 (6B tokens), 400K vocabulary
- **Google Analogy Test Set**: 19,544 questions (8,869 semantic + 10,675 syntactic)


## Author

Ilya Shaposhnikov | [e-mail](mailto:ilia.a.shaposhnikov@gmail.com) | [LinkedIn](https://linkedin.com/in/iliashaposhnikov)

**[Russian Version / На русском](README.ru.md)**