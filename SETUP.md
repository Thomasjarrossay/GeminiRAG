# GeminiRAG — Guide de configuration

## 1. Obtenir vos clés API

### 🔑 Google AI Studio (Gemini Embedding 2)

**Lien direct :** https://aistudio.google.com/apikey

Étapes :
1. Ouvrir le lien ci-dessus
2. Cliquer **"Create API Key"**
3. Sélectionner ou créer un projet Google Cloud
4. Copier la clé générée
5. Coller dans `.env` → `GOOGLE_API_KEY=sk-...`

**Modèles utilisés :**
- `gemini-embedding-001` — Texte & Images (2 048 tokens max, 3 072 dimensions)
- Pour les **vidéos** : extraction de frames → embedding image par frame → moyenne

> **Note :** Le modèle multimodal complet `gemini-embedding-2-preview` est disponible
> sur Vertex AI (Google Cloud). Pour l'activer, vous aurez besoin d'un compte GCP
> avec l'API Vertex AI activée (région us-central1).

---

### 🗄️ Qdrant (Base vectorielle)

#### Option A — Cloud (recommandé)
1. Créer un compte sur https://cloud.qdrant.io
2. Créer un cluster gratuit (1 Go inclus)
3. Dans le dashboard : copier **Cluster URL** et **API Key**
4. Coller dans `.env` :
   ```
   QDRANT_URL=https://votre-cluster.cloud.qdrant.io:6333
   QDRANT_API_KEY=votre_cle_qdrant
   ```

#### Option B — Local (Docker)
```bash
docker run -p 6333:6333 qdrant/qdrant
```
Puis dans `.env` :
```
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=   # laisser vide
```

---

### 🤖 OpenRouter (LLM pour les réponses)

**Lien direct :** https://openrouter.ai/keys

Étapes :
1. Créer un compte sur https://openrouter.ai
2. Aller dans **Keys** → **Create Key**
3. Copier la clé
4. Coller dans `.env` → `OPENROUTER_API_KEY=sk-or-...`

**Modèles recommandés** (définir dans `OPENROUTER_MODEL`) :
| Modèle | Qualité | Vitesse | Coût |
|--------|---------|---------|------|
| `anthropic/claude-opus-4-6` | ⭐⭐⭐⭐⭐ | Modéré | $$ |
| `anthropic/claude-sonnet-4-6` | ⭐⭐⭐⭐ | Rapide | $ |
| `google/gemini-2.0-flash-001` | ⭐⭐⭐⭐ | Très rapide | $ |
| `openai/gpt-4o` | ⭐⭐⭐⭐ | Rapide | $$ |

---

## 2. Installation

```bash
# Cloner / naviguer dans le dossier
cd GeminiRAG

# Créer un environnement virtuel
python3 -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows

# Installer les dépendances
pip install -r requirements.txt

# Copier et remplir le .env
cp .env.example .env
# Éditer .env avec vos clés
```

---

## 3. Lancement

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Puis ouvrez : **http://localhost:8000**

---

## 4. Utilisation

### Indexer des fichiers
1. **Vidéos** — Glissez un `.mp4`, `.mov` dans la zone "Vidéos"
   - Les frames sont extraites automatiquement
   - Vidéo ≤32s → 1 image/seconde | Vidéo >32s → 32 frames uniformément
2. **Images** — Glissez un `.jpg`, `.png` dans la zone "Images"
3. **Textes/PDF** — Glissez un `.txt`, `.pdf`, `.md` dans la zone "Textes & PDF"

### Interroger la base
1. Tapez votre question dans la barre de chat
2. Filtrez optionnellement par type (texte/image/vidéo/pdf)
3. Appuyez sur **Ctrl+Entrée** ou cliquez l'icône d'envoi
4. Les sources citées apparaissent dans le panneau de droite

---

## 5. Architecture technique

```
Question → embed_text (Gemini) → search Qdrant (cosine) → top-5 chunks
        → prompt + contexte → OpenRouter LLM → réponse + sources
```

**Dimensions des vecteurs :** 768 (configurable via `EMBEDDING_DIM`)
**Métrique :** Cosine similarity (recommandée par Google pour Gemini Embedding 2)
**Modèle embedding :** `gemini-embedding-001` (stable) / `gemini-embedding-2-preview` (Vertex AI)

---

## 6. Fichiers de données

```
data/
├── videos/   → Vidéos déposées via l'interface
├── images/   → Images déposées via l'interface
└── texts/    → Textes/PDFs déposés via l'interface
```
