"""
Visualization module for UMAP dimensionality reduction and plotting.
Creates visualizations of text embeddings and optimal paths.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm
import umap
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform


def reduce_dimensions(embeddings, n_components=2, metric='cosine',
                     n_neighbors=4, min_dist=0.1, random_state=42):
    """
    Reduce embeddings to 2D using UMAP.

    Args:
        embeddings: numpy array of embeddings
        n_components: Number of dimensions to reduce to
        metric: Distance metric
        n_neighbors: Number of neighbors for UMAP
        min_dist: Minimum distance for UMAP
        random_state: Random seed for reproducibility

    Returns:
        numpy array of reduced embeddings
    """
    print(f"Reducing {embeddings.shape[1]}D embeddings to {n_components}D using UMAP...")

    reducer = umap.UMAP(
        n_components=n_components,
        metric=metric,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state
    )

    coords_2d = reducer.fit_transform(embeddings)
    print(f"UMAP reduction complete: {coords_2d.shape}")

    return coords_2d


def plot_path_visualization(coords_2d, path, distance_matrix, texts,
                            title="Optimal Reading Order", output_file=None):
    """
    Create main visualization with path and labels.

    Args:
        coords_2d: 2D coordinates from UMAP
        path: Ordered list of text indices
        distance_matrix: Matrix of pairwise distances
        texts: List of text dictionaries
        title: Plot title
        output_file: Path to save figure (if None, display only)
    """
    fig, ax = plt.subplots(figsize=(16, 12))

    n = len(texts)

    # Draw path connections first (so they're behind points)
    for i in range(len(path) - 1):
        idx1, idx2 = path[i], path[i + 1]
        ax.plot([coords_2d[idx1, 0], coords_2d[idx2, 0]],
                [coords_2d[idx1, 1], coords_2d[idx2, 1]],
                'k-', linewidth=2.5, alpha=0.5, zorder=1)

        # Add distance annotation
        mid_x = (coords_2d[idx1, 0] + coords_2d[idx2, 0]) / 2
        mid_y = (coords_2d[idx1, 1] + coords_2d[idx2, 1]) / 2
        dist = distance_matrix[idx1][idx2]
        ax.text(mid_x, mid_y, f'{dist:.3f}',
                fontsize=9, ha='center',
                bbox=dict(boxstyle='round,pad=0.4',
                         facecolor='white', alpha=0.8, edgecolor='gray'),
                zorder=2)

    # Create color map based on path order
    colors = [path.index(i) for i in range(n)]

    # Plot points
    scatter = ax.scatter(coords_2d[:, 0], coords_2d[:, 1],
                        s=400, c=colors, cmap='viridis',
                        edgecolors='black', linewidth=2.5, zorder=3,
                        alpha=0.9)

    # Add text labels
    for i, (x, y) in enumerate(coords_2d):
        # Position number on the point
        ax.text(x, y, f'{path.index(i) + 1}',
                fontsize=12, ha='center', va='center',
                weight='bold', color='white', zorder=4)

        # Position title below the point
        ax.text(x, y - 0.05, f'\n{texts[i]["title"]}',
                fontsize=10, ha='center', va='top',
                weight='bold', zorder=4,
                bbox=dict(boxstyle='round,pad=0.3',
                         facecolor='lightyellow', alpha=0.7))

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, label='Reading Order')
    cbar.set_label('Reading Order', fontsize=12, weight='bold')

    ax.set_title(title, fontsize=16, weight='bold', pad=20)
    ax.set_xlabel('UMAP Dimension 1', fontsize=12)
    ax.set_ylabel('UMAP Dimension 2', fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {output_file}")

    return fig


def plot_distance_heatmap(distance_matrix, texts, output_file=None):
    """
    Create heatmap of distance matrix.

    Args:
        distance_matrix: Matrix of pairwise distances
        texts: List of text dictionaries
        output_file: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    labels = [t['title'] for t in texts]

    # Create heatmap
    im = ax.imshow(distance_matrix, cmap='RdYlGn_r', aspect='auto')

    # Set ticks and labels
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Cosine Distance', fontsize=12, weight='bold')

    # Add text annotations
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j:
                text = ax.text(j, i, f'{distance_matrix[i, j]:.3f}',
                             ha='center', va='center',
                             color='black' if distance_matrix[i, j] > 0.5 else 'white',
                             fontsize=8)

    ax.set_title('Pairwise Distance Matrix (Cosine Distance)',
                fontsize=14, weight='bold', pad=20)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved heatmap to {output_file}")

    return fig


def plot_dendrogram(distance_matrix, texts, output_file=None):
    """
    Create hierarchical clustering dendrogram.

    Args:
        distance_matrix: Matrix of pairwise distances
        texts: List of text dictionaries
        output_file: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    labels = [t['title'] for t in texts]

    # Ensure diagonal is exactly zero for squareform
    dist_matrix_copy = distance_matrix.copy()
    np.fill_diagonal(dist_matrix_copy, 0)

    # Convert distance matrix to condensed form for linkage
    condensed_dist = squareform(dist_matrix_copy)

    # Perform hierarchical clustering
    linkage_matrix = linkage(condensed_dist, method='average')

    # Create dendrogram
    dendrogram(linkage_matrix, labels=labels, ax=ax,
              orientation='right', color_threshold=0)

    ax.set_title('Hierarchical Clustering of Texts (Average Linkage)',
                fontsize=14, weight='bold')
    ax.set_xlabel('Cosine Distance', fontsize=12)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved dendrogram to {output_file}")

    return fig


def plot_path_comparison(coords_2d, optimal_path, chronological_path,
                        distance_matrix, texts, output_file=None):
    """
    Create side-by-side comparison of optimal vs chronological ordering.

    Args:
        coords_2d: 2D coordinates from UMAP
        optimal_path: Optimal path from algorithm
        chronological_path: Chronological ordering
        distance_matrix: Matrix of pairwise distances
        texts: List of text dictionaries
        output_file: Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))

    def plot_single_path(ax, path, title):
        n = len(texts)

        # Draw path
        for i in range(len(path) - 1):
            idx1, idx2 = path[i], path[i + 1]
            ax.plot([coords_2d[idx1, 0], coords_2d[idx2, 0]],
                   [coords_2d[idx1, 1], coords_2d[idx2, 1]],
                   'k-', linewidth=2.5, alpha=0.5, zorder=1)

        # Color by order in path
        colors = [path.index(i) for i in range(n)]

        # Plot points
        scatter = ax.scatter(coords_2d[:, 0], coords_2d[:, 1],
                           s=400, c=colors, cmap='viridis',
                           edgecolors='black', linewidth=2.5, zorder=3)

        # Add labels
        for i, (x, y) in enumerate(coords_2d):
            ax.text(x, y, f'{path.index(i) + 1}',
                   fontsize=11, ha='center', va='center',
                   weight='bold', color='white', zorder=4)

        ax.set_title(title, fontsize=14, weight='bold')
        ax.grid(True, alpha=0.3)
        return scatter

    # Plot chronological order
    from src.distance import compute_path_distance
    chrono_dist = compute_path_distance(chronological_path, distance_matrix)
    plot_single_path(ax1, chronological_path,
                    f'Chronological Order\nTotal Distance: {chrono_dist:.4f}')

    # Plot optimal order
    optimal_dist = compute_path_distance(optimal_path, distance_matrix)
    plot_single_path(ax2, optimal_path,
                    f'Optimal Order\nTotal Distance: {optimal_dist:.4f}')

    improvement = ((chrono_dist - optimal_dist) / chrono_dist) * 100
    fig.suptitle(f'Path Comparison (Improvement: {improvement:.1f}%)',
                fontsize=16, weight='bold')

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved comparison to {output_file}")

    return fig


def create_all_visualizations(embeddings, optimal_path, distance_matrix,
                             texts, output_dir='outputs'):
    """
    Create all visualizations and save to output directory.

    Args:
        embeddings: Text embeddings
        optimal_path: Optimal reading order
        distance_matrix: Distance matrix
        texts: List of text dictionaries
        output_dir: Directory to save outputs
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Reduce dimensions
    coords_2d = reduce_dimensions(embeddings)

    # Main path visualization
    plot_path_visualization(coords_2d, optimal_path, distance_matrix, texts,
                          title="Optimal Reading Order (Embedding-Based)",
                          output_file=f"{output_dir}/optimal_path.png")

    # Distance heatmap
    plot_distance_heatmap(distance_matrix, texts,
                         output_file=f"{output_dir}/distance_matrix.png")

    # Dendrogram
    plot_dendrogram(distance_matrix, texts,
                   output_file=f"{output_dir}/dendrogram.png")

    # Comparison with chronological order
    chronological_path = [i for i in range(len(texts))]
    plot_path_comparison(coords_2d, optimal_path, chronological_path,
                        distance_matrix, texts,
                        output_file=f"{output_dir}/comparison.png")

    print(f"\nAll visualizations saved to {output_dir}/")

    return coords_2d
