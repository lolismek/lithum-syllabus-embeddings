# Embedding-Based Syllabus Reordering

A modular pipeline that optimizes the reading order of ancient texts using semantic similarity embeddings. This project demonstrates how machine learning can help organize curriculum content based on conceptual relationships rather than chronology alone.

## Overview

This system analyzes 9 foundational ancient texts and finds an optimal reading order that minimizes conceptual distance between consecutive texts. It uses:

- **Sentence-BERT embeddings** to capture semantic content
- **Cosine distance** to measure similarity between texts
- **Dynamic programming (Held-Karp)** to find the optimal Hamiltonian path
- **UMAP** for 2D visualization of the semantic space

## Project Structure

```
lithum_embeddings/
├── requirements.txt           # Python dependencies
├── data/
│   └── texts.py              # Text descriptions and metadata
├── src/
│   ├── embeddings.py         # Embedding generation & caching
│   ├── distance.py           # Distance matrix computation
│   ├── pathfinding.py        # Optimization algorithms
│   ├── visualization.py      # UMAP & plotting
│   └── utils.py              # Helper functions
├── main.py                   # Orchestration script
├── outputs/                  # Generated visualizations & reports
└── cache/                    # Cached embeddings
```

## Installation

1. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Find the globally optimal reading order:
```bash
python main.py
```

### Starting with a Specific Text

You can constrain the optimization to start with a particular text:

```bash
# List all available texts
python main.py --list

# Start with a specific text
python main.py --start "Iliad"
python main.py --start "Genesis"
python main.py --start "Gospel of Mark"
```

### Command-Line Options

- `--list` - Display all available texts and exit
- `--start "Title"` - Optimize reading order starting with the specified text (case-sensitive)
- `--help` - Show help message

### What the Pipeline Does

The pipeline will:
1. Load the 9 ancient texts
2. Generate semantic embeddings (cached for reuse)
3. Compute pairwise distance matrix
4. Find optimal reading order using multiple algorithms
5. Generate visualizations and reports

When using `--start`, the algorithms find the optimal path that **must begin** with your specified text, which may result in a different (slightly longer) total path distance compared to the globally optimal solution.

## Results

The pipeline generates:

- **optimal_path.png** - UMAP visualization with optimal reading path
- **distance_matrix.png** - Heatmap of pairwise text similarities
- **dendrogram.png** - Hierarchical clustering tree
- **comparison.png** - Side-by-side optimal vs. chronological ordering
- **results.json** - Machine-readable output
- **report.txt** - Comprehensive text report

### Key Findings

The optimal semantic ordering achieved a **14.4% improvement** over chronological ordering:

- **Chronological distance**: 4.9493
- **Optimal distance**: 4.2355

**Optimal Reading Order:**
1. Gospel of Mark
2. Genesis
3. Epic of Gilgamesh
4. Aeneid
5. Odyssey
6. Iliad
7. Oresteia
8. Sappho Fragments
9. Exaltation of Inanna

The algorithm found that Homer's epics (Iliad & Odyssey) have the smallest semantic distance (0.2683), while religious creation texts cluster together (Mark → Genesis → Gilgamesh).

### Starting with Specific Texts

Different starting texts produce different optimal paths:

**Starting with "Iliad":**
- Total distance: 4.5259 (8.6% improvement)
- Path: Iliad → Odyssey → Aeneid → Gilgamesh → Oresteia → Sappho → Inanna → Genesis → Mark

**Starting with "Genesis":**
- Total distance: 4.3667 (11.8% improvement)
- Path: Genesis → Gilgamesh → Aeneid → Odyssey → Iliad → Oresteia → Sappho → Inanna → Mark

These constrained paths are slightly longer than the globally optimal path (4.2355) but may be pedagogically preferable depending on your curriculum goals.

## Algorithms

Three pathfinding algorithms are implemented:

1. **Greedy Nearest Neighbor** - Fast heuristic (O(n³))
2. **Held-Karp Dynamic Programming** - Exact solution (O(n² · 2ⁿ))
3. **Brute Force** - Verification only for n ≤ 10

## Performance

- Total runtime: **~7 seconds** (meets <10s target)
- Embedding generation: ~3.5s (cached after first run)
- UMAP reduction: ~6s
- Pathfinding: <1s

## Technical Details

- **Model**: `all-MiniLM-L6-v2` (384-dimensional embeddings)
- **Distance metric**: Cosine distance
- **Dimensionality reduction**: UMAP with cosine metric
- **Platform**: CPU-optimized, works on MacBook ARM/Intel

## Extension Ideas

- Compare multiple embedding models
- Add pedagogical constraints (e.g., chronological anchoring)
- Interactive visualization with Plotly
- Sensitivity analysis across different UMAP parameters
- Domain-specific fine-tuning on classical literature

## License

This project is for educational and research purposes.
