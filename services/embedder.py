"""
Gemini Embedding 2 — Service d'embedding multimodal
====================================================
Modèles :
  - gemini-embedding-001 : texte uniquement (2048 tokens max)
  - gemini-embedding-2-preview : texte + image + audio + vidéo (Vertex AI)

Pour l'API Google AI Studio, on utilise gemini-embedding-001 pour texte/image.
Pour la vidéo, on extrait les frames et on moyenne les embeddings image.

Dimensions recommandées : 768 (rapport qualité/performance)
Métrique de distance : cosine similarity
"""

import os
import base64
import logging
import tempfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import google.generativeai as genai
from PIL import Image

logger = logging.getLogger(__name__)

# Modèles Gemini Embedding
MODEL_TEXT = "models/gemini-embedding-exp-03-07"   # Meilleur modèle exp (text)
MODEL_MULTIMODAL = "models/gemini-embedding-001"    # Stable (text + image via parts)
FALLBACK_MODEL = "models/text-embedding-004"        # Fallback léger

# Stratégie vidéo : Matryoshka MRL
# Videos ≤32s → 1 FPS  |  Videos >32s → 32 frames uniformément
MAX_FRAMES = 32
FPS_SHORT_VIDEO = 1
MAX_VIDEO_DURATION_SEC = 120  # Limite Gemini Embedding 2

TASK_TYPE_MAP = {
    "text": "RETRIEVAL_DOCUMENT",
    "query": "RETRIEVAL_QUERY",
    "image": "RETRIEVAL_DOCUMENT",
    "video": "RETRIEVAL_DOCUMENT",
}


class GeminiEmbedder:
    """Service d'embedding multimodal via Gemini Embedding 2."""

    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY non défini dans .env")
        genai.configure(api_key=api_key)

        self.dim = int(os.getenv("EMBEDDING_DIM", "768"))
        self.model = MODEL_MULTIMODAL
        logger.info(f"GeminiEmbedder initialisé — modèle: {self.model}, dim: {self.dim}")

    def _embed_raw(self, content, task_type: str = "RETRIEVAL_DOCUMENT") -> list[float]:
        """Appel bas niveau à l'API Gemini Embedding."""
        result = genai.embed_content(
            model=self.model,
            content=content,
            task_type=task_type,
            output_dimensionality=self.dim,
        )
        return result["embedding"]

    def embed_text(self, text: str, is_query: bool = False) -> list[float]:
        """Embed un texte brut."""
        task = "RETRIEVAL_QUERY" if is_query else "RETRIEVAL_DOCUMENT"
        return self._embed_raw(text, task_type=task)

    def embed_image(self, image_path: str) -> list[float]:
        """
        Embed une image (PNG, JPEG).
        Envoie l'image en base64 via les 'parts' de l'API multimodale.
        """
        path = Path(image_path)
        with open(path, "rb") as f:
            image_bytes = f.read()

        # Déterminer le mime type
        suffix = path.suffix.lower()
        mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                    ".png": "image/png", ".webp": "image/webp"}
        mime_type = mime_map.get(suffix, "image/jpeg")

        # L'API Gemini accepte dict avec inline_data pour les images
        content = {
            "parts": [
                {
                    "inline_data": {
                        "mime_type": mime_type,
                        "data": base64.b64encode(image_bytes).decode("utf-8"),
                    }
                }
            ]
        }
        return self._embed_raw(content)

    def embed_video(self, video_path: str) -> list[float]:
        """
        Embed une vidéo par extraction de frames.

        Stratégie (conforme à la doc Gemini Embedding 2) :
        - Vidéo ≤32s : échantillonnage à 1 FPS
        - Vidéo >32s : 32 frames uniformément répartis
        - Max 120 secondes par segment
        - Averaging des embeddings de frames → vecteur agrégé unique
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Impossible d'ouvrir la vidéo : {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / fps

        logger.info(f"Vidéo: {Path(video_path).name} | {duration_sec:.1f}s | {fps:.1f}fps")

        # Calcul des indices de frames à extraire
        if duration_sec <= 32:
            # 1 FPS pour les vidéos courtes
            frame_indices = [int(i * fps) for i in range(int(duration_sec * FPS_SHORT_VIDEO))]
        else:
            # 32 frames uniformément répartis pour les vidéos longues
            frame_indices = [int(i * total_frames / MAX_FRAMES) for i in range(MAX_FRAMES)]

        frame_indices = [min(idx, total_frames - 1) for idx in frame_indices]
        frame_indices = list(dict.fromkeys(frame_indices))  # déduplication

        embeddings = []
        with tempfile.TemporaryDirectory() as tmp_dir:
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    continue

                # Sauvegarder la frame temporairement
                frame_path = os.path.join(tmp_dir, f"frame_{idx}.jpg")
                cv2.imwrite(frame_path, frame)

                try:
                    emb = self.embed_image(frame_path)
                    embeddings.append(emb)
                except Exception as e:
                    logger.warning(f"Frame {idx} ignorée : {e}")

        cap.release()

        if not embeddings:
            raise ValueError("Aucune frame n'a pu être embedée")

        # Agrégation par moyenne (Matryoshka MRL compatible)
        avg_embedding = np.mean(embeddings, axis=0).tolist()
        logger.info(f"Vidéo embedée : {len(embeddings)} frames → vecteur dim {len(avg_embedding)}")
        return avg_embedding

    def embed_text_chunks(self, chunks: list[str]) -> list[list[float]]:
        """Embed plusieurs chunks de texte en batch."""
        return [self.embed_text(chunk) for chunk in chunks]
