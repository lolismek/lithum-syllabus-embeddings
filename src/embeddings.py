"""
Embedding generation and caching module.
Uses sentence-transformers to generate semantic embeddings for texts.
"""

import os
import hashlib
import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = 'all-MiniLM-L6-v2'
CACHE_DIR = 'cache'


def get_cache_filename(model_name, texts):
    """
    Generate deterministic cache filename based on model and text content.

    Args:
        model_name: Name of the embedding model
        texts: List of text dictionaries

    Returns:
        Path to cache file
    """
    # Create hash of text content for cache invalidation
    content_hash = hashlib.md5(
        str([(t['title'], t['description']) for t in texts]).encode()
    ).hexdigest()[:8]

    filename = f"embeddings_{model_name.replace('/', '_')}_{content_hash}.npy"
    return os.path.join(CACHE_DIR, filename)


def load_model(model_name=MODEL_NAME):
    """
    Load the sentence transformer model.

    Args:
        model_name: Name of the model to load

    Returns:
        SentenceTransformer model
    """
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    return model


def prepare_text_for_embedding(text_dict):
    """
    Prepare text for embedding by concatenating title and description.

    Args:
        text_dict: Dictionary with 'title' and 'description' keys

    Returns:
        Combined string for embedding
    """
    return f"{text_dict['title']}. {text_dict['description']}"


def generate_embeddings(texts, model_name=MODEL_NAME):
    """
    Generate embeddings for a list of texts.

    Args:
        texts: List of text dictionaries
        model_name: Name of the model to use

    Returns:
        numpy array of shape (n_texts, embedding_dim)
    """
    model = load_model(model_name)

    # Prepare texts
    combined_texts = [prepare_text_for_embedding(t) for t in texts]

    print(f"Generating embeddings for {len(combined_texts)} texts...")
    embeddings = model.encode(combined_texts, show_progress_bar=True)

    return np.array(embeddings)


def save_embeddings(embeddings, cache_file):
    """
    Save embeddings to cache file.

    Args:
        embeddings: numpy array of embeddings
        cache_file: Path to cache file
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.save(cache_file, embeddings)
    print(f"Saved embeddings to {cache_file}")


def load_embeddings(cache_file):
    """
    Load embeddings from cache file.

    Args:
        cache_file: Path to cache file

    Returns:
        numpy array of embeddings
    """
    embeddings = np.load(cache_file)
    print(f"Loaded embeddings from {cache_file}")
    return embeddings


def get_or_generate_embeddings(texts, model_name=MODEL_NAME, force_regenerate=False):
    """
    Get embeddings from cache or generate if not cached.

    Args:
        texts: List of text dictionaries
        model_name: Name of the model to use
        force_regenerate: If True, ignore cache and regenerate

    Returns:
        numpy array of embeddings
    """
    cache_file = get_cache_filename(model_name, texts)

    if not force_regenerate and os.path.exists(cache_file):
        print("Found cached embeddings")
        return load_embeddings(cache_file)
    else:
        embeddings = generate_embeddings(texts, model_name)
        save_embeddings(embeddings, cache_file)
        return embeddings


def get_embedding_info(embeddings):
    """
    Get information about embeddings.

    Args:
        embeddings: numpy array of embeddings

    Returns:
        Dictionary with embedding statistics
    """
    return {
        'shape': embeddings.shape,
        'dtype': embeddings.dtype,
        'n_texts': embeddings.shape[0],
        'embedding_dim': embeddings.shape[1],
        'mean_norm': np.mean(np.linalg.norm(embeddings, axis=1)),
        'std_norm': np.std(np.linalg.norm(embeddings, axis=1))
    }
