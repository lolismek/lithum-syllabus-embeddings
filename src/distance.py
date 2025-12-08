"""
Distance matrix computation module.
Computes pairwise cosine distances between text embeddings.
"""

import numpy as np
from scipy.spatial.distance import cdist, cosine


def compute_distance_matrix(embeddings, metric='cosine'):
    """
    Compute pairwise distance matrix using cosine distance.

    Args:
        embeddings: numpy array of shape (n_texts, embedding_dim)
        metric: Distance metric to use (default: 'cosine')

    Returns:
        numpy array of shape (n_texts, n_texts) containing pairwise distances
    """
    print(f"Computing {metric} distance matrix...")

    # Use vectorized cdist for efficiency
    distance_matrix = cdist(embeddings, embeddings, metric=metric)

    # Verify matrix properties
    verify_distance_matrix(distance_matrix)

    return distance_matrix


def verify_distance_matrix(distance_matrix, tolerance=1e-10):
    """
    Verify distance matrix properties.

    Args:
        distance_matrix: numpy array to verify
        tolerance: Tolerance for numerical comparisons

    Raises:
        AssertionError if verification fails
    """
    n = distance_matrix.shape[0]

    # Check square matrix
    assert distance_matrix.shape == (n, n), \
        f"Matrix must be square, got {distance_matrix.shape}"

    # Check symmetry
    max_asymmetry = np.max(np.abs(distance_matrix - distance_matrix.T))
    assert max_asymmetry < tolerance, \
        f"Matrix not symmetric, max asymmetry: {max_asymmetry}"

    # Check zero diagonal
    max_diagonal = np.max(np.abs(np.diag(distance_matrix)))
    assert max_diagonal < tolerance, \
        f"Diagonal must be zero, max value: {max_diagonal}"

    # Check non-negative
    min_value = np.min(distance_matrix)
    assert min_value >= -tolerance, \
        f"All distances must be non-negative, min value: {min_value}"

    print("Distance matrix verification passed")


def get_distance_statistics(distance_matrix):
    """
    Compute statistics about the distance matrix.

    Args:
        distance_matrix: numpy array of pairwise distances

    Returns:
        Dictionary with distance statistics
    """
    # Get upper triangle (excluding diagonal)
    n = distance_matrix.shape[0]
    upper_triangle = distance_matrix[np.triu_indices(n, k=1)]

    return {
        'mean': np.mean(upper_triangle),
        'std': np.std(upper_triangle),
        'min': np.min(upper_triangle),
        'max': np.max(upper_triangle),
        'median': np.median(upper_triangle),
        'n_comparisons': len(upper_triangle)
    }


def get_nearest_neighbors(distance_matrix, k=3):
    """
    Find k nearest neighbors for each text.

    Args:
        distance_matrix: numpy array of pairwise distances
        k: Number of nearest neighbors to find

    Returns:
        List of tuples (text_idx, [(neighbor_idx, distance), ...])
    """
    n = distance_matrix.shape[0]
    neighbors = []

    for i in range(n):
        # Get distances to all other texts
        distances = [(j, distance_matrix[i, j]) for j in range(n) if i != j]
        # Sort by distance and take top k
        distances.sort(key=lambda x: x[1])
        neighbors.append((i, distances[:k]))

    return neighbors


def compute_path_distance(path, distance_matrix):
    """
    Compute total distance for a given path through the texts.

    Args:
        path: List of text indices representing the order
        distance_matrix: numpy array of pairwise distances

    Returns:
        Total distance of the path
    """
    total_distance = 0.0
    for i in range(len(path) - 1):
        total_distance += distance_matrix[path[i], path[i + 1]]
    return total_distance


def compare_paths(path1, path2, distance_matrix):
    """
    Compare two paths and return their distances.

    Args:
        path1: First path (list of indices)
        path2: Second path (list of indices)
        distance_matrix: Distance matrix

    Returns:
        Dictionary with comparison results
    """
    dist1 = compute_path_distance(path1, distance_matrix)
    dist2 = compute_path_distance(path2, distance_matrix)

    improvement = ((dist1 - dist2) / dist1) * 100 if dist1 > 0 else 0

    return {
        'path1_distance': dist1,
        'path2_distance': dist2,
        'difference': dist1 - dist2,
        'improvement_percent': improvement,
        'better_path': 1 if dist1 < dist2 else 2
    }
