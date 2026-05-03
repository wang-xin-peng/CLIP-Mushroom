import faiss
import numpy as np
import torch
import pickle
from pathlib import Path


class FaissIndex:
    """FAISS vector index for fast similarity search.

    Supports cosine similarity (via inner product on normalized vectors)
    and stores metadata for each indexed vector.
    """

    def __init__(self, dim=512, metric="cosine"):
        if metric == "cosine":
            self.index = faiss.IndexFlatIP(dim)  # Inner product = cosine for normalized vectors
        else:
            self.index = faiss.IndexFlatL2(dim)
        self.metadata = []

    def add(self, features, metadata):
        """Add (N, D) numpy features and corresponding metadata list.

        Args:
            features: (N, D) float32 numpy array
            metadata: list of dicts with "path" and "label" keys
        """
        if isinstance(features, torch.Tensor):
            features = features.cpu().numpy()
        features = np.asarray(features, dtype=np.float32)
        self.index.add(features)
        self.metadata.extend(metadata)

    def search(self, query_vector, k=10):
        """Search for top-k nearest neighbors.

        Args:
            query_vector: (D,) or (1, D) tensor or numpy array
            k: number of results

        Returns:
            list of dicts with "path", "label", and "score" keys
        """
        if isinstance(query_vector, torch.Tensor):
            query_vector = query_vector.cpu().numpy()
        query_vector = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        scores, indices = self.index.search(query_vector, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.metadata):
                results.append({**self.metadata[idx], "score": float(score)})
            else:
                results.append({"path": "", "label": "unknown", "score": float(score)})
        return results

    def save(self, path):
        """Save FAISS index and metadata to disk."""
        path = Path(path)
        faiss.write_index(self.index, str(path))
        with open(str(path) + ".meta.pkl", "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self, path):
        """Load FAISS index and metadata from disk."""
        path = Path(path)
        self.index = faiss.read_index(str(path))
        with open(str(path) + ".meta.pkl", "rb") as f:
            self.metadata = pickle.load(f)
