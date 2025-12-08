"""
Pathfinding algorithms for optimal text ordering.
Solves the shortest Hamiltonian path problem.
"""

import numpy as np
from itertools import permutations


def greedy_nearest_neighbor(distance_matrix, start_idx=None):
    """
    Greedy nearest neighbor algorithm starting from a specific index.

    Args:
        distance_matrix: numpy array of pairwise distances
        start_idx: Starting index (if None, will try all starts)

    Returns:
        Tuple of (path, total_distance)
    """
    n = len(distance_matrix)

    if start_idx is None:
        # Try all starting points and return the best
        return greedy_nearest_neighbor_all_starts(distance_matrix)

    path = [start_idx]
    unvisited = set(range(n)) - {start_idx}
    current = start_idx
    total_distance = 0.0

    while unvisited:
        # Find nearest unvisited node
        nearest = min(unvisited, key=lambda x: distance_matrix[current][x])
        total_distance += distance_matrix[current][nearest]
        path.append(nearest)
        current = nearest
        unvisited.remove(nearest)

    return path, total_distance


def greedy_nearest_neighbor_all_starts(distance_matrix):
    """
    Try greedy nearest neighbor from all possible starting points.

    Args:
        distance_matrix: numpy array of pairwise distances

    Returns:
        Tuple of (best_path, best_distance)
    """
    n = len(distance_matrix)
    best_path = None
    best_distance = float('inf')

    print(f"Running greedy nearest neighbor from all {n} starting points...")

    for start_idx in range(n):
        path, distance = greedy_nearest_neighbor(distance_matrix, start_idx)

        if distance < best_distance:
            best_distance = distance
            best_path = path

    print(f"Greedy algorithm found path with distance: {best_distance:.4f}")
    return best_path, best_distance


def held_karp_path(distance_matrix, start_idx=None):
    """
    Held-Karp dynamic programming algorithm for shortest Hamiltonian path.
    Finds the exact optimal solution.

    Args:
        distance_matrix: numpy array of pairwise distances
        start_idx: Starting index (if None, tries all starts)

    Returns:
        Tuple of (optimal_path, optimal_distance)
    """
    n = len(distance_matrix)

    if start_idx is not None:
        print(f"Running Held-Karp DP algorithm with fixed start (n={n}, start={start_idx})...")
    else:
        print(f"Running Held-Karp DP algorithm for exact solution (n={n})...")

    # dp[mask][i] = (min_distance, previous_node)
    # mask represents the set of visited nodes as a bitmask
    # i is the current ending node
    dp = {}

    # Initialize: single node paths
    if start_idx is not None:
        # Only initialize with the specified starting node
        mask = 1 << start_idx
        dp[(mask, start_idx)] = (0.0, None)
    else:
        # Try all possible starting nodes
        for i in range(n):
            mask = 1 << i
            dp[(mask, i)] = (0.0, None)

    # Build up subsets of increasing size
    for subset_size in range(2, n + 1):
        for mask in range(1 << n):
            if bin(mask).count('1') != subset_size:
                continue

            # Try all possible ending nodes in this subset
            for i in range(n):
                if not (mask & (1 << i)):
                    continue

                # Remove i from the mask
                prev_mask = mask ^ (1 << i)

                # Try all possible previous nodes
                min_dist = float('inf')
                best_prev = None

                for j in range(n):
                    if not (prev_mask & (1 << j)):
                        continue

                    # Distance = distance to reach j + distance from j to i
                    prev_dist, _ = dp.get((prev_mask, j), (float('inf'), None))
                    total_dist = prev_dist + distance_matrix[j][i]

                    if total_dist < min_dist:
                        min_dist = total_dist
                        best_prev = j

                if best_prev is not None:
                    dp[(mask, i)] = (min_dist, best_prev)

    # Find the best ending node
    full_mask = (1 << n) - 1
    min_dist = float('inf')
    best_end = None

    for i in range(n):
        dist, _ = dp.get((full_mask, i), (float('inf'), None))
        if dist < min_dist:
            min_dist = dist
            best_end = i

    # Reconstruct path
    path = []
    mask = full_mask
    current = best_end

    while current is not None:
        path.append(current)
        prev_mask = mask ^ (1 << current)
        _, prev = dp.get((mask, current), (None, None))
        mask = prev_mask
        current = prev

    path.reverse()

    print(f"Held-Karp algorithm found optimal path with distance: {min_dist:.4f}")
    return path, min_dist


def brute_force_path(distance_matrix, start_idx=None):
    """
    Brute force solution: try all permutations.
    Only practical for very small n (n <= 10).

    Args:
        distance_matrix: numpy array of pairwise distances
        start_idx: Starting index (if None, tries all starts)

    Returns:
        Tuple of (optimal_path, optimal_distance)
    """
    n = len(distance_matrix)

    if n > 10:
        raise ValueError("Brute force only practical for n <= 10")

    if start_idx is not None:
        # Only try permutations starting with start_idx
        remaining = [i for i in range(n) if i != start_idx]
        num_perms = np.math.factorial(n - 1)
        print(f"Running brute force with fixed start on {num_perms} permutations...")

        best_path = None
        best_distance = float('inf')

        for perm in permutations(remaining):
            full_perm = [start_idx] + list(perm)
            distance = 0.0
            for i in range(n - 1):
                distance += distance_matrix[full_perm[i]][full_perm[i + 1]]

            if distance < best_distance:
                best_distance = distance
                best_path = full_perm
    else:
        print(f"Running brute force on all {np.math.factorial(n)} permutations...")

        best_path = None
        best_distance = float('inf')

        for perm in permutations(range(n)):
            distance = 0.0
            for i in range(n - 1):
                distance += distance_matrix[perm[i]][perm[i + 1]]

            if distance < best_distance:
                best_distance = distance
                best_path = list(perm)

    print(f"Brute force found optimal path with distance: {best_distance:.4f}")
    return best_path, best_distance


def compare_algorithms(distance_matrix, start_idx=None):
    """
    Compare different pathfinding algorithms.

    Args:
        distance_matrix: numpy array of pairwise distances
        start_idx: Starting index (if None, tries all starts)

    Returns:
        Dictionary with results from each algorithm
    """
    results = {}

    # Greedy
    if start_idx is not None:
        greedy_path, greedy_dist = greedy_nearest_neighbor(distance_matrix, start_idx)
        results['greedy'] = {
            'path': greedy_path,
            'distance': greedy_dist,
            'algorithm': f'Greedy Nearest Neighbor (start={start_idx})'
        }
    else:
        greedy_path, greedy_dist = greedy_nearest_neighbor_all_starts(distance_matrix)
        results['greedy'] = {
            'path': greedy_path,
            'distance': greedy_dist,
            'algorithm': 'Greedy Nearest Neighbor'
        }

    # Held-Karp (exact)
    hk_path, hk_dist = held_karp_path(distance_matrix, start_idx)
    if start_idx is not None:
        results['held_karp'] = {
            'path': hk_path,
            'distance': hk_dist,
            'algorithm': f'Held-Karp DP (Exact, start={start_idx})'
        }
    else:
        results['held_karp'] = {
            'path': hk_path,
            'distance': hk_dist,
            'algorithm': 'Held-Karp DP (Exact)'
        }

    # Brute force if small enough
    n = len(distance_matrix)
    if n <= 10:
        bf_path, bf_dist = brute_force_path(distance_matrix, start_idx)
        if start_idx is not None:
            results['brute_force'] = {
                'path': bf_path,
                'distance': bf_dist,
                'algorithm': f'Brute Force (start={start_idx})'
            }
        else:
            results['brute_force'] = {
                'path': bf_path,
                'distance': bf_dist,
                'algorithm': 'Brute Force'
            }

    return results


def verify_path(path, n):
    """
    Verify that a path is valid (visits all nodes exactly once).

    Args:
        path: List of node indices
        n: Total number of nodes

    Returns:
        Boolean indicating validity
    """
    return (
        len(path) == n and
        len(set(path)) == n and
        all(0 <= node < n for node in path)
    )
