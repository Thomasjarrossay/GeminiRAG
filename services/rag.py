"""
RAG Pipeline — Retrieval-Augmented Generation via OpenRouter
============================================================
1. Embed la question utilisateur (Gemini)
2. Recherche les chunks similaires dans Qdrant
3. Construit le prompt contextuel
4. Interroge le LLM via OpenRouter
5. Retourne la réponse avec les sources citées

Variables .env requises :
  OPENROUTER_API_KEY : clé OpenRouter
  OPENROUTER_MODEL   : modèle (ex: anthropic/claude-opus-4-6)
"""

import os
import logging
import httpx

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

SYSTEM_PROMPT = """Tu es un assistant intelligent qui répond aux questions basées sur les documents fournis.

Règles :
- Base tes réponses UNIQUEMENT sur le contexte fourni
- Si l'information n'est pas dans le contexte, dis-le clairement
- Cite tes sources en indiquant le fichier source entre crochets [nom_fichier]
- Sois précis, concis et utile
- Pour les vidéos et images, décris ce qu'elles contiennent selon les informations disponibles
- Réponds en français sauf si la question est posée dans une autre langue
"""


def build_context(search_results: list[dict]) -> str:
    """Construit le contexte RAG depuis les résultats de recherche."""
    if not search_results:
        return "Aucun document pertinent trouvé dans la base de connaissances."

    context_parts = []
    for i, r in enumerate(search_results, 1):
        score = r.get("score", 0)
        source = r.get("source_file", "inconnu")
        file_type = r.get("file_type", "")
        content = r.get("content", "")

        type_label = {
            "text": "📄 Texte",
            "pdf": "📋 PDF",
            "image": "🖼️ Image",
            "video": "🎬 Vidéo",
        }.get(file_type, "📎 Fichier")

        context_parts.append(
            f"[Source {i}] {type_label} — {source} (score: {score:.3f})\n{content}"
        )

    return "\n\n---\n\n".join(context_parts)


async def query_llm(
    question: str,
    search_results: list[dict],
    model: Optional[str] = None,
) -> dict:
    """
    Interroge le LLM via OpenRouter avec le contexte RAG.

    Returns:
        {
            "answer": str,
            "model": str,
            "sources": list[dict],
            "tokens_used": int
        }
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY non défini dans .env")

    llm_model = model or os.getenv("OPENROUTER_MODEL", "anthropic/claude-opus-4-6")
    context = build_context(search_results)

    user_message = f"""Contexte de la base de connaissances :
{context}

---

Question : {question}"""

    payload = {
        "model": llm_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0.3,
        "max_tokens": 2048,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://geminirag.local",
        "X-Title": "GeminiRAG",
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            OPENROUTER_BASE_URL,
            json=payload,
            headers=headers,
        )
        response.raise_for_status()
        data = response.json()

    answer = data["choices"][0]["message"]["content"]
    tokens = data.get("usage", {}).get("total_tokens", 0)

    logger.info(f"LLM réponse — modèle: {llm_model} | tokens: {tokens}")

    return {
        "answer": answer,
        "model": llm_model,
        "sources": search_results,
        "tokens_used": tokens,
    }


# Correction: import Optional manquant
from typing import Optional
