"""
Utility functions for reporting and output formatting.
"""

import json
import os
from datetime import datetime


def print_header(text, char='='):
    """Print formatted header."""
    print(f"\n{char * len(text)}")
    print(text)
    print(f"{char * len(text)}\n")


def print_optimal_path(path, distance_matrix, texts):
    """
    Print the optimal reading order to terminal.

    Args:
        path: Ordered list of text indices
        distance_matrix: Matrix of pairwise distances
        texts: List of text dictionaries
    """
    print_header("OPTIMAL READING ORDER")

    total_distance = 0.0

    for i, idx in enumerate(path):
        text = texts[idx]
        print(f"{i + 1}. {text['title']}")
        print(f"   Date: {text['approx_date']} {'BCE' if text['approx_date'] < 0 else 'CE'}")

        if i < len(path) - 1:
            next_idx = path[i + 1]
            dist = distance_matrix[idx][next_idx]
            total_distance += dist
            print(f"   → distance to next: {dist:.4f}")
        print()

    print(f"TOTAL PATH DISTANCE: {total_distance:.4f}\n")


def print_comparison(optimal_path, chronological_path, distance_matrix, texts):
    """
    Print comparison between optimal and chronological ordering.

    Args:
        optimal_path: Optimal path from algorithm
        chronological_path: Chronological ordering
        distance_matrix: Distance matrix
        texts: List of text dictionaries
    """
    from src.distance import compute_path_distance

    optimal_dist = compute_path_distance(optimal_path, distance_matrix)
    chrono_dist = compute_path_distance(chronological_path, distance_matrix)
    improvement = ((chrono_dist - optimal_dist) / chrono_dist) * 100

    print_header("COMPARISON WITH CHRONOLOGICAL ORDER")

    print(f"Chronological total distance: {chrono_dist:.4f}")
    print(f"Optimal total distance:       {optimal_dist:.4f}")
    print(f"Improvement:                  {improvement:.1f}%\n")


def print_transition_analysis(path, distance_matrix, texts):
    """
    Print analysis of individual transitions.

    Args:
        path: Ordered list of text indices
        distance_matrix: Distance matrix
        texts: List of text dictionaries
    """
    print_header("INDIVIDUAL TRANSITIONS", '-')

    transitions = []
    for i in range(len(path) - 1):
        idx1, idx2 = path[i], path[i + 1]
        dist = distance_matrix[idx1][idx2]
        transitions.append((
            texts[idx1]['title'],
            texts[idx2]['title'],
            dist
        ))

    # Sort by distance
    transitions.sort(key=lambda x: x[2])

    print("Smallest jumps:")
    for title1, title2, dist in transitions[:3]:
        print(f"  {title1} → {title2}: {dist:.4f}")

    print("\nLargest jumps:")
    for title1, title2, dist in transitions[-3:]:
        print(f"  {title1} → {title2}: {dist:.4f}")

    print()


def print_algorithm_comparison(results):
    """
    Print comparison of different algorithms.

    Args:
        results: Dictionary with results from different algorithms
    """
    print_header("ALGORITHM COMPARISON", '-')

    for name, result in results.items():
        print(f"{result['algorithm']}:")
        print(f"  Distance: {result['distance']:.4f}")
        print(f"  Path: {result['path']}\n")


def save_results_json(optimal_path, distance_matrix, texts, results,
                     output_file='outputs/results.json'):
    """
    Save results to JSON file.

    Args:
        optimal_path: Optimal reading order
        distance_matrix: Distance matrix
        texts: List of text dictionaries
        results: Algorithm comparison results
        output_file: Path to output file
    """
    from src.distance import compute_path_distance

    chronological_path = list(range(len(texts)))
    optimal_dist = compute_path_distance(optimal_path, distance_matrix)
    chrono_dist = compute_path_distance(chronological_path, distance_matrix)

    output = {
        'timestamp': datetime.now().isoformat(),
        'optimal_path': {
            'order': optimal_path,
            'distance': optimal_dist,
            'texts': [texts[i]['title'] for i in optimal_path]
        },
        'chronological_path': {
            'order': chronological_path,
            'distance': chrono_dist,
            'texts': [texts[i]['title'] for i in chronological_path]
        },
        'improvement_percent': ((chrono_dist - optimal_dist) / chrono_dist) * 100,
        'algorithms': {
            name: {
                'distance': result['distance'],
                'path': result['path']
            }
            for name, result in results.items()
        },
        'distance_matrix': distance_matrix.tolist(),
        'texts': texts
    }

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Saved results to {output_file}")


def generate_text_report(optimal_path, distance_matrix, texts, results,
                         embeddings_info, output_file='outputs/report.txt'):
    """
    Generate comprehensive text report.

    Args:
        optimal_path: Optimal reading order
        distance_matrix: Distance matrix
        texts: List of text dictionaries
        results: Algorithm comparison results
        embeddings_info: Information about embeddings
        output_file: Path to output file
    """
    from src.distance import compute_path_distance, get_distance_statistics

    chronological_path = list(range(len(texts)))
    optimal_dist = compute_path_distance(optimal_path, distance_matrix)
    chrono_dist = compute_path_distance(chronological_path, distance_matrix)
    dist_stats = get_distance_statistics(distance_matrix)

    lines = []
    lines.append("=" * 80)
    lines.append("EMBEDDING-BASED SYLLABUS REORDERING - COMPREHENSIVE REPORT")
    lines.append("=" * 80)
    lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    lines.append("\n" + "=" * 80)
    lines.append("EMBEDDING INFORMATION")
    lines.append("=" * 80)
    for key, value in embeddings_info.items():
        lines.append(f"{key}: {value}")

    lines.append("\n" + "=" * 80)
    lines.append("DISTANCE MATRIX STATISTICS")
    lines.append("=" * 80)
    for key, value in dist_stats.items():
        if isinstance(value, float):
            lines.append(f"{key}: {value:.4f}")
        else:
            lines.append(f"{key}: {value}")

    lines.append("\n" + "=" * 80)
    lines.append("OPTIMAL READING ORDER")
    lines.append("=" * 80)

    for i, idx in enumerate(optimal_path):
        text = texts[idx]
        lines.append(f"\n{i + 1}. {text['title']}")
        lines.append(f"   Date: {text['approx_date']} {'BCE' if text['approx_date'] < 0 else 'CE'}")
        lines.append(f"   Chronological position: {text['chronological_order']}")

        if i < len(optimal_path) - 1:
            next_idx = optimal_path[i + 1]
            dist = distance_matrix[idx][next_idx]
            lines.append(f"   → distance to next: {dist:.4f}")

    lines.append(f"\nTOTAL PATH DISTANCE: {optimal_dist:.4f}")

    lines.append("\n" + "=" * 80)
    lines.append("COMPARISON WITH CHRONOLOGICAL ORDER")
    lines.append("=" * 80)
    lines.append(f"Chronological total distance: {chrono_dist:.4f}")
    lines.append(f"Optimal total distance:       {optimal_dist:.4f}")
    improvement = ((chrono_dist - optimal_dist) / chrono_dist) * 100
    lines.append(f"Improvement:                  {improvement:.1f}%")

    lines.append("\n" + "=" * 80)
    lines.append("ALGORITHM COMPARISON")
    lines.append("=" * 80)
    for name, result in results.items():
        lines.append(f"\n{result['algorithm']}:")
        lines.append(f"  Distance: {result['distance']:.4f}")
        lines.append(f"  Path: {result['path']}")

    lines.append("\n" + "=" * 80)
    lines.append("SAVED ARTIFACTS")
    lines.append("=" * 80)
    lines.append("- outputs/optimal_path.png - Main visualization")
    lines.append("- outputs/distance_matrix.png - Distance heatmap")
    lines.append("- outputs/dendrogram.png - Hierarchical clustering")
    lines.append("- outputs/comparison.png - Optimal vs chronological")
    lines.append("- outputs/results.json - Machine-readable results")
    lines.append("- outputs/report.txt - This report")

    lines.append("\n" + "=" * 80)

    report_text = '\n'.join(lines)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(report_text)

    print(f"Saved report to {output_file}")

    return report_text


def print_saved_artifacts(output_dir='outputs'):
    """Print list of saved artifacts."""
    print_header("SAVED ARTIFACTS", '-')
    print(f"- {output_dir}/optimal_path.png - Main visualization")
    print(f"- {output_dir}/distance_matrix.png - Distance heatmap")
    print(f"- {output_dir}/dendrogram.png - Hierarchical clustering")
    print(f"- {output_dir}/comparison.png - Optimal vs chronological")
    print(f"- {output_dir}/results.json - Machine-readable results")
    print(f"- {output_dir}/report.txt - Full analysis")
    print()
