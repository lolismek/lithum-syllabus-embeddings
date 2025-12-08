#!/usr/bin/env python3
"""
Main orchestration script for embedding-based syllabus reordering.
Coordinates the entire pipeline from text loading to visualization.
"""

import time
import sys
import argparse
from data.texts import get_texts
from src.embeddings import get_or_generate_embeddings, get_embedding_info
from src.distance import compute_distance_matrix, get_distance_statistics
from src.pathfinding import (
    greedy_nearest_neighbor_all_starts,
    held_karp_path,
    compare_algorithms,
    verify_path
)
from src.visualization import create_all_visualizations
from src.utils import (
    print_header,
    print_optimal_path,
    print_comparison,
    print_transition_analysis,
    print_algorithm_comparison,
    save_results_json,
    generate_text_report,
    print_saved_artifacts
)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Optimize reading order for ancient texts using semantic similarity.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                           # Find globally optimal order
  python main.py --start "Iliad"           # Start with Iliad
  python main.py --start "Genesis"         # Start with Genesis
  python main.py --list                    # List all available texts
        """
    )

    parser.add_argument(
        '--start',
        type=str,
        help='Title of the text to start with (case-sensitive)'
    )

    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available texts and exit'
    )

    return parser.parse_args()


def find_text_index(texts, title):
    """
    Find the index of a text by title.

    Args:
        texts: List of text dictionaries
        title: Title to search for

    Returns:
        Index of the text, or None if not found
    """
    for i, text in enumerate(texts):
        if text['title'] == title:
            return i
    return None


def list_texts(texts):
    """Print list of available texts."""
    print_header("AVAILABLE TEXTS")
    for i, text in enumerate(texts):
        date_str = f"{abs(text['approx_date'])} {'BCE' if text['approx_date'] < 0 else 'CE'}"
        print(f"{i}. {text['title']:30s} ({date_str})")
    print()


def main():
    """Main execution pipeline."""
    # Parse arguments
    args = parse_arguments()

    # Load texts first (needed for --list)
    texts = get_texts()

    # Handle --list flag
    if args.list:
        list_texts(texts)
        return 0

    # Handle --start argument
    start_idx = None
    if args.start:
        start_idx = find_text_index(texts, args.start)
        if start_idx is None:
            print(f"ERROR: Text '{args.start}' not found.")
            print("\nAvailable texts:")
            list_texts(texts)
            return 1

    start_time = time.time()

    print_header("EMBEDDING-BASED SYLLABUS REORDERING")
    if start_idx is not None:
        print(f"Optimizing reading order starting with: {texts[start_idx]['title']}\n")
    else:
        print("Finding globally optimal reading order for ancient texts\n")

    # Phase 1: Load texts
    print_header("Phase 1: Loading Texts", '-')
    phase_start = time.time()
    print(f"Loaded {len(texts)} texts")
    if start_idx is not None:
        print(f"Starting text: {texts[start_idx]['title']} (index {start_idx})")
    print(f"Time: {time.time() - phase_start:.3f}s\n")

    # Phase 2: Generate embeddings
    print_header("Phase 2: Generating Embeddings", '-')
    phase_start = time.time()
    embeddings = get_or_generate_embeddings(texts)
    embeddings_info = get_embedding_info(embeddings)
    print(f"Embeddings shape: {embeddings_info['shape']}")
    print(f"Embedding dimension: {embeddings_info['embedding_dim']}")
    print(f"Time: {time.time() - phase_start:.3f}s\n")

    # Phase 3: Compute distance matrix
    print_header("Phase 3: Computing Distance Matrix", '-')
    phase_start = time.time()
    distance_matrix = compute_distance_matrix(embeddings)
    dist_stats = get_distance_statistics(distance_matrix)
    print(f"Distance statistics:")
    print(f"  Mean: {dist_stats['mean']:.4f}")
    print(f"  Std:  {dist_stats['std']:.4f}")
    print(f"  Min:  {dist_stats['min']:.4f}")
    print(f"  Max:  {dist_stats['max']:.4f}")
    print(f"Time: {time.time() - phase_start:.3f}s\n")

    # Phase 4: Find optimal paths
    print_header("Phase 4: Finding Optimal Paths", '-')
    phase_start = time.time()

    # Compare all algorithms
    results = compare_algorithms(distance_matrix, start_idx)

    # Verify paths
    for name, result in results.items():
        is_valid = verify_path(result['path'], len(texts))
        print(f"{result['algorithm']}: {'Valid' if is_valid else 'INVALID'}")

        # Verify starting text if specified
        if start_idx is not None and result['path'][0] != start_idx:
            print(f"  WARNING: Path does not start with specified text!")

    print(f"Time: {time.time() - phase_start:.3f}s\n")

    # Use Held-Karp (exact) result as the optimal path
    optimal_path = results['held_karp']['path']
    optimal_distance = results['held_karp']['distance']

    # Phase 5: Generate visualizations
    print_header("Phase 5: Generating Visualizations", '-')
    phase_start = time.time()
    create_all_visualizations(embeddings, optimal_path, distance_matrix, texts)
    print(f"Time: {time.time() - phase_start:.3f}s\n")

    # Phase 6: Generate reports
    print_header("Phase 6: Creating Reports", '-')
    phase_start = time.time()

    # Print to terminal
    print_optimal_path(optimal_path, distance_matrix, texts)

    chronological_path = list(range(len(texts)))
    print_comparison(optimal_path, chronological_path, distance_matrix, texts)
    print_transition_analysis(optimal_path, distance_matrix, texts)
    print_algorithm_comparison(results)

    # Save outputs
    save_results_json(optimal_path, distance_matrix, texts, results)
    generate_text_report(optimal_path, distance_matrix, texts, results, embeddings_info)

    print_saved_artifacts()

    print(f"Time: {time.time() - phase_start:.3f}s\n")

    # Final summary
    total_time = time.time() - start_time
    print_header("EXECUTION COMPLETE")
    print(f"Total runtime: {total_time:.2f}s")

    # Verify performance constraint
    if total_time > 10:
        print(f"WARNING: Runtime exceeded 10s target (actual: {total_time:.2f}s)")
    else:
        print(f"Performance target met (<10s)")

    print("\nAll artifacts saved to outputs/")
    print("Run completed successfully!\n")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
