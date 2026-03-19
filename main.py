"""
GeminiRAG — Backend FastAPI
============================
RAG multimodal avec Gemini Embedding 2 + Qdrant + OpenRouter.

Endpoints :
  POST /api/ingest        : Upload et indexation d'un fichier
  POST /api/query         : Question → réponse RAG
  GET  /api/documents     : Liste des documents indexés
  DELETE /api/documents   : Supprime un document

Lancement :
  uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import os
import uuid
import logging
import shutil
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Chargement des variables d'environnement
load_dotenv()

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# Répertoire de données
DATA_DIR = Path("data")
UPLOAD_TEMP_DIR = Path("uploads_temp")
UPLOAD_TEMP_DIR.mkdir(exist_ok=True)

# Services (initialisés au démarrage)
embedder = None
vector_store = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialisation des services au démarrage."""
    global embedder, vector_store
    logger.info("🚀 Démarrage GeminiRAG...")

    try:
        from services.embedder import GeminiEmbedder
        from services.vector_store import QdrantStore

        embedder = GeminiEmbedder()
        vector_store = QdrantStore()
        logger.info("✅ Services initialisés")
    except Exception as e:
        logger.error(f"❌ Erreur initialisation : {e}")
        raise

    yield

    logger.info("GeminiRAG arrêté.")


app = FastAPI(
    title="GeminiRAG",
    description="RAG multimodal — Gemini Embedding 2 + Qdrant + OpenRouter",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS pour le frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Fichiers statiques (frontend)
app.mount("/static", StaticFiles(directory="static"), name="static")


# ─────────────────────────────────────────────
# Modèles Pydantic
# ─────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    file_type_filter: Optional[str] = None
    model: Optional[str] = None


class DeleteRequest(BaseModel):
    source_file: str


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.get("/")
async def root():
    """Sert l'interface web principale."""
    return FileResponse("static/index.html")


@app.get("/api/health")
async def health():
    """Health check."""
    try:
        count = vector_store.count() if vector_store else 0
        return {
            "status": "ok",
            "vectors_indexed": count,
            "embedding_dim": int(os.getenv("EMBEDDING_DIM", "768")),
            "model": "gemini-embedding-001",
        }
    except Exception as e:
        return JSONResponse(status_code=503, content={"status": "error", "message": str(e)})


@app.post("/api/ingest")
async def ingest_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Upload et indexation d'un fichier.
    Supporte : txt, md, pdf, jpg, jpeg, png, mp4, mov, avi, webm.
    """
    from services.ingestion import process_file, detect_file_type, SUPPORTED_TEXT, SUPPORTED_IMAGE, SUPPORTED_VIDEO, SUPPORTED_PDF

    allowed_extensions = SUPPORTED_TEXT | SUPPORTED_IMAGE | SUPPORTED_VIDEO | SUPPORTED_PDF
    suffix = Path(file.filename).suffix.lower()

    if suffix not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Type de fichier non supporté : {suffix}. Acceptés : {', '.join(sorted(allowed_extensions))}",
        )

    # Sauvegarder le fichier temporairement
    temp_path = UPLOAD_TEMP_DIR / f"{uuid.uuid4()}{suffix}"
    try:
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Copier dans le bon dossier data/
        file_type = detect_file_type(str(temp_path))
        dest_dir = DATA_DIR / f"{file_type}s"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / file.filename
        shutil.copy2(temp_path, dest_path)

        # Traitement et indexation
        chunks, detected_type = process_file(str(dest_path))
        if not chunks:
            raise HTTPException(status_code=422, detail="Impossible de traiter ce fichier")

        # Embedding selon le type
        points = []
        for chunk in chunks:
            try:
                if detected_type in ("text", "pdf"):
                    vector = embedder.embed_text(chunk.content)
                elif detected_type == "image":
                    vector = embedder.embed_image(str(dest_path))
                elif detected_type == "video":
                    vector = embedder.embed_video(str(dest_path))
                else:
                    vector = embedder.embed_text(chunk.content)

                points.append({
                    "id": str(uuid.uuid4()),
                    "vector": vector,
                    "payload": {
                        "content": chunk.content,
                        "source_file": chunk.source_file,
                        "file_type": chunk.file_type,
                        "chunk_index": chunk.chunk_index,
                        "metadata": chunk.metadata,
                    },
                })
            except Exception as e:
                logger.warning(f"Chunk {chunk.chunk_index} ignoré : {e}")
                continue

        if not points:
            raise HTTPException(status_code=422, detail="Aucun vecteur généré")

        indexed_count = vector_store.upsert(points)

        return {
            "status": "success",
            "file": file.filename,
            "file_type": detected_type,
            "chunks_indexed": indexed_count,
            "total_vectors": vector_store.count(),
        }

    finally:
        if temp_path.exists():
            temp_path.unlink()


@app.post("/api/query")
async def query_rag(request: QueryRequest):
    """
    Pose une question au RAG.
    Recherche les chunks pertinents et génère une réponse via OpenRouter.
    """
    from services.rag import query_llm

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="La question ne peut pas être vide")

    # 1. Embed la question
    query_vector = embedder.embed_text(request.question, is_query=True)

    # 2. Recherche dans Qdrant
    search_results = vector_store.search(
        query_vector=query_vector,
        limit=request.top_k,
        file_type_filter=request.file_type_filter,
    )

    # 3. Génération de la réponse
    result = await query_llm(
        question=request.question,
        search_results=search_results,
        model=request.model,
    )

    return result


@app.get("/api/documents")
async def list_documents():
    """Liste tous les documents indexés."""
    docs = vector_store.list_documents()
    total = vector_store.count()
    return {
        "documents": docs,
        "total_documents": len(docs),
        "total_vectors": total,
    }


@app.delete("/api/documents")
async def delete_document(request: DeleteRequest):
    """Supprime un document et tous ses vecteurs."""
    vector_store.delete_document(request.source_file)
    return {
        "status": "deleted",
        "source_file": request.source_file,
        "total_vectors": vector_store.count(),
    }


@app.get("/api/stats")
async def get_stats():
    """Statistiques de la base vectorielle."""
    docs = vector_store.list_documents()
    by_type = {}
    for d in docs:
        ft = d["file_type"]
        if ft not in by_type:
            by_type[ft] = {"count": 0, "chunks": 0}
        by_type[ft]["count"] += 1
        by_type[ft]["chunks"] += d["chunk_count"]

    return {
        "total_vectors": vector_store.count(),
        "total_documents": len(docs),
        "by_type": by_type,
        "embedding_model": "gemini-embedding-001",
        "embedding_dim": int(os.getenv("EMBEDDING_DIM", "768")),
        "llm_model": os.getenv("OPENROUTER_MODEL", ""),
    }
