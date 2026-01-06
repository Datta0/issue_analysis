import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
import re
from typing import List, Dict, Any, Optional

class SimilarityEngine:
    def index(self, documents: List[Dict[str, Any]]):
        raise NotImplementedError

    def find_similar(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        raise NotImplementedError

class HybridEngine(SimilarityEngine):
    def __init__(self, model_name: str = 'BAAI/bge-base-en-v1.5', alpha: float = 0.5, device: str = None, batch_size: int = 32):
        """
        Args:
            model_name: SentenceTransformer model name.
            alpha: Weight for dense score (0.0 to 1.0).
            device: 'cuda', 'cpu', or None (auto-detect).
            batch_size: Batch size for encoding.
        """
        self.model_name = model_name
        self.alpha = alpha

        # Auto-detect device if not specified
        if not device:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        print(f"Initializing HybridEngine with model='{model_name}' on device='{self.device}'...")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.batch_size = batch_size

        self.documents = []
        self.corpus_embeddings = None
        self.bm25 = None

    def _preprocess(self, text: str) -> str:
        # Simple preprocessing
        return re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())

    def index(self, documents: List[Dict[str, Any]]):
        """
        Index a list of documents.
        """
        self.documents = documents

        corpus_text = []
        tokenized_corpus = []

        print(f"Indexing {len(documents)} documents (Batch Size: {self.batch_size})...")

        for doc in documents:
            # Title + Body for context
            full_text = f"{doc.get('title', '')} {doc.get('body', '')}"
            corpus_text.append(full_text)

            tokenized_corpus.append(self._preprocess(full_text).split())

        # 1. Compute Dense Embeddings (Batched)
        self.corpus_embeddings = self.model.encode(
            corpus_text,
            convert_to_tensor=True,
            device=self.device,
            batch_size=self.batch_size,
            show_progress_bar=True
        )

        # 2. Build BM25 Index
        self.bm25 = BM25Okapi(tokenized_corpus)

        print("Indexing complete.")

    def find_similar(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if not self.documents:
            return []

        # --- Dense Retrieval ---
        query_embedding = self.model.encode(query, convert_to_tensor=True, device=self.device)

        # Compute cosine similarity
        # self.corpus_embeddings is on GPU if device=cuda, match query to it
        dense_scores = util.cos_sim(query_embedding, self.corpus_embeddings)[0].cpu().numpy()

        # --- BM25 Retrieval ---
        tokenized_query = self._preprocess(query).split()
        bm25_scores = np.array(self.bm25.get_scores(tokenized_query))

        # --- Normalization ---
        if bm25_scores.max() > 0:
            bm25_scores = bm25_scores / bm25_scores.max()

        dense_scores = np.maximum(dense_scores, 0)

        # --- Hybrid Score Fusion ---
        hybrid_scores = (self.alpha * dense_scores) + ((1 - self.alpha) * bm25_scores)

        # Get top-k indices
        top_indices = np.argsort(hybrid_scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = hybrid_scores[idx]
            doc = self.documents[idx]
            results.append({
                'score': float(score),
                'number': doc['number'],
                'title': doc['title'],
                'url': doc['html_url']
            })

        return results

    def save_cache(self, path: str):
        """Save the index and documents to disk."""
        import pickle
        if not self.documents:
             return

        cache_data = {
            'documents': self.documents,
            'corpus_embeddings': self.corpus_embeddings,
            'bm25': self.bm25
        }
        with open(path, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"Saved similarity cache to {path}")

    def load_cache(self, path: str):
        """Load the index from disk."""
        import pickle
        import os
        if not os.path.exists(path):
            return False

        try:
            with open(path, 'rb') as f:
                cache_data = pickle.load(f)

            self.documents = cache_data['documents']
            self.corpus_embeddings = cache_data['corpus_embeddings']
            self.bm25 = cache_data['bm25']
            print(f"Loaded similarity cache from {path} ({len(self.documents)} docs)")
            return True
        except Exception as e:
            print(f"Failed to load cache: {e}")
            return False
