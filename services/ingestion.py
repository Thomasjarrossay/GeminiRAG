"""
Ingestion multimodale — Traitement des fichiers
===============================================
Gère le traitement de :
  - Texte : .txt, .md, .pdf, .docx → chunks de 500 tokens
  - Images : .jpg, .jpeg, .png, .webp
  - Vidéos : .mp4, .mov, .mpeg, .avi

Retourne des chunks normalisés prêts à être embedés et indexés.
"""

import os
import re
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

SUPPORTED_TEXT = {".txt", ".md", ".csv", ".py", ".js", ".json", ".html"}
SUPPORTED_IMAGE = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
SUPPORTED_VIDEO = {".mp4", ".mov", ".mpeg", ".avi", ".mkv", ".webm"}
SUPPORTED_PDF = {".pdf"}

CHUNK_SIZE = 800     # Caractères par chunk
CHUNK_OVERLAP = 100  # Chevauchement entre chunks


@dataclass
class Chunk:
    """Unité d'indexation."""
    content: str           # Texte du chunk / description
    source_file: str       # Nom du fichier source
    file_type: str         # "text" | "image" | "video" | "pdf"
    chunk_index: int = 0
    metadata: dict = field(default_factory=dict)


def detect_file_type(path: str) -> str:
    """Détecte le type de fichier par son extension."""
    suffix = Path(path).suffix.lower()
    if suffix in SUPPORTED_TEXT:
        return "text"
    if suffix in SUPPORTED_IMAGE:
        return "image"
    if suffix in SUPPORTED_VIDEO:
        return "video"
    if suffix in SUPPORTED_PDF:
        return "pdf"
    return "unknown"


def split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Découpe un texte en chunks avec chevauchement.
    Essaie de couper sur des phrases complètes.
    """
    # Nettoyer les espaces excessifs
    text = re.sub(r"\n{3,}", "\n\n", text.strip())

    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunks.append(text[start:])
            break

        # Essayer de couper à la dernière phrase
        cut = text.rfind(". ", start, end)
        if cut == -1 or cut <= start:
            cut = text.rfind("\n", start, end)
        if cut == -1 or cut <= start:
            cut = end

        chunks.append(text[start:cut + 1].strip())
        start = max(cut - overlap, start + 1)

    return [c for c in chunks if c.strip()]


def process_text_file(file_path: str) -> list[Chunk]:
    """Traite un fichier texte → liste de chunks."""
    path = Path(file_path)
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except Exception as e:
        logger.error(f"Erreur lecture {file_path}: {e}")
        return []

    text_chunks = split_text(content)
    return [
        Chunk(
            content=chunk,
            source_file=path.name,
            file_type="text",
            chunk_index=i,
            metadata={"file_size": path.stat().st_size, "extension": path.suffix},
        )
        for i, chunk in enumerate(text_chunks)
    ]


def process_pdf_file(file_path: str) -> list[Chunk]:
    """Traite un PDF → extraction texte via pypdf."""
    path = Path(file_path)
    try:
        import pypdf
        reader = pypdf.PdfReader(str(path))
        text_parts = []
        for i, page in enumerate(reader.pages):
            t = page.extract_text()
            if t:
                text_parts.append(f"[Page {i+1}]\n{t}")
        full_text = "\n\n".join(text_parts)
    except ImportError:
        logger.warning("pypdf non installé, traitement PDF désactivé")
        return []
    except Exception as e:
        logger.error(f"Erreur PDF {file_path}: {e}")
        return []

    text_chunks = split_text(full_text)
    return [
        Chunk(
            content=chunk,
            source_file=path.name,
            file_type="pdf",
            chunk_index=i,
            metadata={"pages": len(reader.pages)},
        )
        for i, chunk in enumerate(text_chunks)
    ]


def process_image(file_path: str) -> list[Chunk]:
    """
    Traite une image → 1 chunk avec description basique.
    Le vecteur sera calculé depuis les pixels (embed_image).
    On stocke une description textuelle pour la citation.
    """
    path = Path(file_path)
    try:
        from PIL import Image
        img = Image.open(str(path))
        width, height = img.size
        mode = img.mode
        description = f"Image: {path.name} ({width}x{height}, {mode})"
    except Exception:
        description = f"Image: {path.name}"

    return [
        Chunk(
            content=description,
            source_file=path.name,
            file_type="image",
            chunk_index=0,
            metadata={"path": str(path)},
        )
    ]


def process_video(file_path: str) -> list[Chunk]:
    """
    Traite une vidéo → 1 chunk agrégé.
    Le vecteur sera calculé par extraction de frames (embed_video).
    """
    path = Path(file_path)
    try:
        import cv2
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frames / fps if fps > 0 else 0
        cap.release()
        description = f"Vidéo: {path.name} | Durée: {duration:.1f}s | FPS: {fps:.1f}"
    except Exception:
        description = f"Vidéo: {path.name}"

    return [
        Chunk(
            content=description,
            source_file=path.name,
            file_type="video",
            chunk_index=0,
            metadata={"path": str(path)},
        )
    ]


def process_file(file_path: str) -> tuple[list[Chunk], str]:
    """
    Point d'entrée principal.
    Retourne (chunks, file_type).
    """
    file_type = detect_file_type(file_path)
    suffix = Path(file_path).suffix.lower()

    if file_type == "text":
        return process_text_file(file_path), "text"
    elif file_type == "pdf":
        return process_pdf_file(file_path), "pdf"
    elif file_type == "image":
        return process_image(file_path), "image"
    elif file_type == "video":
        return process_video(file_path), "video"
    else:
        logger.warning(f"Type de fichier non supporté : {suffix}")
        return [], "unknown"
