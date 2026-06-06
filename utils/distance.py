import numpy as np
from typing import List


def euclidean_distance(vector1: List[float], vector2: List[float]) -> float:
    """
    Calculate Euclidean distance between two vectors
    """
    v1 = np.array(vector1)
    v2 = np.array(vector2)
    return float(np.linalg.norm(v1 - v2))

def cosine_similarity(vector1: List[float], vector2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors
    """
    v1 = np.array(vector1)
    v2 = np.array(vector2)
    
    dot_product = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(dot_product / (norm1 * norm2))

def cosine_distance(vector1: List[float], vector2: List[float]) -> float:
    """
    Calculate cosine distance between two vectors
    """
    return 1 - cosine_similarity(vector1, vector2)

def calculate_distance(vector1: List[float], vector2: List[float], 
                      metric: str = 'cosine') -> float:
    """
    Calculate distance between two vectors using specified metric
    """
    if metric == 'euclidean':
        return euclidean_distance(vector1, vector2)
    elif metric == 'cosine':
        return cosine_distance(vector1, vector2)
    else:
        raise ValueError(f"Unsupported distance metric: {metric}")


# ============================================================
# Vectorized batch distance functions
# These replace per-element Python loops with single NumPy ops.
# A search over N neighbors goes from N python calls to 1 matmul.
# ============================================================

def batch_cosine_distance(vectors: np.ndarray, query: np.ndarray) -> np.ndarray:
    """
    Compute cosine distance between a query and multiple vectors in one shot.

    For pre-normalized vectors (norm=1), this is simply ``1 - vectors @ query``.
    Falls back to full normalization when vectors are not unit-length.

    Args:
        vectors: (N, D) array of candidate vectors
        query:   (D,) query vector

    Returns:
        (N,) array of cosine distances in [0, 2]
    """
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)

    # Dot products: (N,)
    dots = vectors @ query

    # Norms
    query_norm = np.linalg.norm(query)
    if query_norm < 1e-10:
        return np.ones(vectors.shape[0], dtype=np.float32)

    vec_norms = np.linalg.norm(vectors, axis=1)
    # Guard against zero-norm vectors
    vec_norms = np.maximum(vec_norms, 1e-10)

    similarities = dots / (vec_norms * query_norm)
    return (1.0 - similarities).astype(np.float32)


def batch_cosine_distance_normalized(vectors: np.ndarray, query: np.ndarray) -> np.ndarray:
    """
    Fast path for pre-normalized vectors (HNSW always normalizes at insert).
    Skips norm computation entirely — just ``1 - dot``.

    Args:
        vectors: (N, D) array of unit-length vectors
        query:   (D,) unit-length query vector

    Returns:
        (N,) array of cosine distances
    """
    return (1.0 - vectors @ query).astype(np.float32)


def batch_euclidean_distance(vectors: np.ndarray, query: np.ndarray) -> np.ndarray:
    """
    Compute Euclidean distance between a query and multiple vectors.

    Uses the expansion ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a·b
    which avoids materializing the (N, D) difference matrix.

    Args:
        vectors: (N, D) array
        query:   (D,) vector

    Returns:
        (N,) array of L2 distances
    """
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)

    # ||v||^2 for each vector
    vec_sq = np.sum(vectors ** 2, axis=1)
    # ||q||^2 broadcast
    query_sq = np.dot(query, query)
    # -2 * v · q
    cross = vectors @ query

    sq_dists = vec_sq + query_sq - 2.0 * cross
    # Clamp numerical noise to zero
    sq_dists = np.maximum(sq_dists, 0.0)
    return np.sqrt(sq_dists).astype(np.float32)


def batch_dot_product(vectors: np.ndarray, query: np.ndarray) -> np.ndarray:
    """
    Compute negative inner product distance (higher dot = closer = lower distance).

    Args:
        vectors: (N, D) array
        query:   (D,) vector

    Returns:
        (N,) array of negative dot products (lower = more similar)
    """
    return -(vectors @ query).astype(np.float32)
