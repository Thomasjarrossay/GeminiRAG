"""
Chroma Vector Store — Gestion de la base vectorielle
=====================================================
Base embarquée dans le container Docker — aucun service externe requis.
Persistance via volume Docker monté sur /app/data/chroma.

Variables .env utilisées :
  CHROMA_PATH    : Chemin de persistance (défaut: /app/data/chroma)
  CHROMA_COLLECTION: Nom de la collection (défaut: geminirag)
  EMBEDDING_DIM  : Dimension des vecteurs (768 par défaut, info only)
"""

import os
import logging
import uuid
from typing import Optional

import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)


class QdrantStore:
    """
    Interface haut niveau pour Chroma.
    Conserve le nom QdrantStore pour compatibilité avec le reste du code.
    """

    def __init__(self):
        chroma_path = os.getenv("CHROMA_PATH", "/app/data/chroma")
        self.collection_name = os.getenv("CHROMA_COLLECTION", os.getenv("QDRANT_COLLECTION", "geminirag"))
        self.dim = int(os.getenv("EMBEDDING_DIM", "768"))

        os.makedirs(chroma_path, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=chroma_path,
            settings=Settings(anonymized_telemetry=False),
        )

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},  # Gemini → cosine similarity
        )

        logger.info(f"Chroma initialisé | path: {chroma_path} | collection: {self.collection_name}")

    def upsert(self, points: list[dict]) -> int:
        """
        Insère ou met à jour des points.

        Format attendu pour chaque point :
        {
            "id": str (uuid optionnel),
            "vector": list[float],
            "payload": {
                "content": str,
                "source_file": str,
                "file_type": str,     # "text" | "image" | "video" | "pdf"
                "chunk_index": int,
                "metadata": dict,
            }
        }
        """
        ids = []
        embeddings = []
        documents = []
        metadatas = []

        for p in points:
            point_id = p.get("id") or str(uuid.uuid4())
            payload = p.get("payload", {})

            ids.append(point_id)
            embeddings.append(p["vector"])
            documents.append(payload.get("content", ""))
            metadatas.append({
                "source_file": payload.get("source_file", ""),
                "file_type": payload.get("file_type", ""),
                "chunk_index": payload.get("chunk_index", 0),
            })

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        logger.info(f"{len(ids)} points insérés dans '{self.collection_name}'")
        return len(ids)

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
        where = {"file_type": file_type_filter} if file_type_filter else None

        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=min(limit, max(self.collection.count(), 1)),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        output = []
        if not results["ids"] or not results["ids"][0]:
            return output

        for i, doc_id in enumerate(results["ids"][0]):
            distance = results["distances"][0][i]
            # Chroma cosine distance : 0 = identique, 2 = opposé
            # Convertir en score de similarité [0, 1]
            score = 1 - (distance / 2)
            meta = results["metadatas"][0][i] if results["metadatas"] else {}
            output.append({
                "score": round(score, 4),
                "content": results["documents"][0][i] if results["documents"] else "",
                "source_file": meta.get("source_file", ""),
                "file_type": meta.get("file_type", ""),
                "chunk_index": meta.get("chunk_index", 0),
                "metadata": {},
            })

        return output

    def list_documents(self) -> list[dict]:
        """Liste les documents indexés (groupés par source_file)."""
        total = self.collection.count()
        if total == 0:
            return []

        results = self.collection.get(
            include=["metadatas"],
            limit=10000,
        )

        docs = {}
        for meta in results.get("metadatas", []):
            src = meta.get("source_file", "unknown")
            if src not in docs:
                docs[src] = {
                    "source_file": src,
                    "file_type": meta.get("file_type", ""),
                    "chunk_count": 0,
                }
            docs[src]["chunk_count"] += 1

        return list(docs.values())

    def delete_document(self, source_file: str) -> int:
        """Supprime tous les points d'un fichier source."""
        results = self.collection.get(
            where={"source_file": source_file},
            include=[],
        )
        ids_to_delete = results.get("ids", [])
        if ids_to_delete:
            self.collection.delete(ids=ids_to_delete)
        logger.info(f"Document '{source_file}' supprimé ({len(ids_to_delete)} chunks)")
        return len(ids_to_delete)

    def count(self) -> int:
        """Retourne le nombre total de vecteurs indexés."""
        return self.collection.count()
