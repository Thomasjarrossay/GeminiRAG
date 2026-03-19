"""
Qdrant Vector Store — Gestion de la base vectorielle
=====================================================
Compatible Qdrant Cloud et Qdrant local (localhost:6333).

Variables .env requises :
  QDRANT_URL       : URL du cluster Qdrant
  QDRANT_API_KEY   : Clé API (optionnelle pour local)
  QDRANT_COLLECTION: Nom de la collection
  EMBEDDING_DIM    : Dimension des vecteurs (768 par défaut)
"""

import os
import logging
import uuid
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    SearchRequest,
)

logger = logging.getLogger(__name__)


class QdrantStore:
    """Interface haut niveau pour Qdrant."""

    def __init__(self):
        url = os.getenv("QDRANT_URL", "http://localhost:6333")
        api_key = os.getenv("QDRANT_API_KEY") or None
        self.collection = os.getenv("QDRANT_COLLECTION", "geminirag")
        self.dim = int(os.getenv("EMBEDDING_DIM", "768"))

        self.client = QdrantClient(url=url, api_key=api_key, timeout=30)
        logger.info(f"Qdrant connecté : {url} | collection: {self.collection}")

        self._ensure_collection()

    def _ensure_collection(self):
        """Crée la collection si elle n'existe pas."""
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection not in existing:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=self.dim,
                    distance=Distance.COSINE,  # Gemini → cosine similarity
                ),
            )
            logger.info(f"Collection '{self.collection}' créée (dim={self.dim})")
        else:
            logger.info(f"Collection '{self.collection}' déjà existante")

    def upsert(self, points: list[dict]) -> int:
        """
        Insère ou met à jour des points.

        Format attendu pour chaque point :
        {
            "id": str (uuid optionnel),
            "vector": list[float],
            "payload": {
                "content": str,       # texte/description
                "source_file": str,   # nom du fichier source
                "file_type": str,     # "text" | "image" | "video"
                "chunk_index": int,   # index du chunk dans le fichier
                "metadata": dict,     # métadonnées supplémentaires
            }
        }
        """
        qdrant_points = []
        for p in points:
            point_id = p.get("id") or str(uuid.uuid4())
            # Qdrant accepte les UUIDs comme string
            qdrant_points.append(
                PointStruct(
                    id=point_id,
                    vector=p["vector"],
                    payload=p.get("payload", {}),
                )
            )

        self.client.upsert(
            collection_name=self.collection,
            points=qdrant_points,
        )
        logger.info(f"{len(qdrant_points)} points insérés dans '{self.collection}'")
        return len(qdrant_points)

    def search(
        self,
        query_vector: list[float],
        limit: int = 5,
        file_type_filter: Optional[str] = None,
    ) -> list[dict]:
        """
        Recherche les points les plus proches par similarité cosine.

        Returns:
            Liste de dicts {score, content, source_file, file_type, metadata}
        """
        query_filter = None
        if file_type_filter:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="file_type",
                        match=MatchValue(value=file_type_filter),
                    )
                ]
            )

        results = self.client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
        )

        return [
            {
                "score": r.score,
                "content": r.payload.get("content", ""),
                "source_file": r.payload.get("source_file", ""),
                "file_type": r.payload.get("file_type", ""),
                "chunk_index": r.payload.get("chunk_index", 0),
                "metadata": r.payload.get("metadata", {}),
            }
            for r in results
        ]

    def list_documents(self) -> list[dict]:
        """Liste les documents indexés (groupés par source_file)."""
        # Scroll sur tous les points pour récupérer les fichiers uniques
        points, _ = self.client.scroll(
            collection_name=self.collection,
            limit=1000,
            with_payload=True,
            with_vectors=False,
        )

        docs = {}
        for p in points:
            src = p.payload.get("source_file", "unknown")
            if src not in docs:
                docs[src] = {
                    "source_file": src,
                    "file_type": p.payload.get("file_type", ""),
                    "chunk_count": 0,
                }
            docs[src]["chunk_count"] += 1

        return list(docs.values())

    def delete_document(self, source_file: str) -> int:
        """Supprime tous les points d'un fichier source."""
        result = self.client.delete(
            collection_name=self.collection,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="source_file",
                        match=MatchValue(value=source_file),
                    )
                ]
            ),
        )
        logger.info(f"Document '{source_file}' supprimé")
        return 1

    def count(self) -> int:
        """Retourne le nombre total de vecteurs indexés."""
        info = self.client.get_collection(self.collection)
        return info.points_count or 0
