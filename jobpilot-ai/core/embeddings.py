"""
JobPilot AI - Embeddings Module
Semantic vector embeddings using sentence-transformers.
Used for intelligent job-resume matching (not just keyword matching).
"""

from typing import List, Optional, Union
import numpy as np
from loguru import logger

from config.settings import settings


class EmbeddingEngine:
    """
    Wraps sentence-transformers for generating semantic embeddings.

    These embeddings capture MEANING, not just keywords.
    So "Python developer" matches "Software Engineer skilled in Python"
    even without exact word overlap.
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.embedding_model
        self._model = None   # Lazy load

    @property
    def model(self):
        """Lazy load the model to avoid startup cost."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            logger.success(f"Embedding model loaded: {self.model_name}")
        return self._model

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Encode text(s) into vector embeddings.

        Args:
            texts: Single string or list of strings

        Returns:
            numpy array of shape (n_texts, embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(texts, show_progress_bar=False)

    def cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.
        Returns value between -1 and 1 (higher = more similar).
        """
        from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
        a = vec_a.reshape(1, -1)
        b = vec_b.reshape(1, -1)
        return float(sk_cosine(a, b)[0][0])

    def batch_similarity(
        self,
        query_vec: np.ndarray,
        corpus_vecs: np.ndarray,
    ) -> np.ndarray:
        """
        Compute cosine similarity of query against all corpus vectors.

        Returns:
            Array of similarity scores, same length as corpus_vecs
        """
        from sklearn.metrics.pairwise import cosine_similarity
        scores = cosine_similarity(query_vec.reshape(1, -1), corpus_vecs)
        return scores[0]

    def semantic_match_score(self, text_a: str, text_b: str) -> float:
        """
        Compute semantic similarity between two texts.
        Used for job-candidate matching.
        """
        vecs = self.encode([text_a, text_b])
        return self.cosine_similarity(vecs[0], vecs[1])


# Global singleton
embedding_engine = EmbeddingEngine()
