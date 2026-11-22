# RAG Prototype (DTUnator, early stage)

Early retrieval-augmented generation prototype (Jan–Feb 2026) for answering DTU students'
doubts from course material. It sits between the hackathon build and v2:

`DTUnator` (BrainWave 2.0 hackathon) → **`rag-prototype`** (this) → `dtunator-rag-sft-rl` (v2: FastAPI app, SFT/RL)

## What's in here

| File | What it does |
|---|---|
| `Python_code_1.py` | **Main pipeline.** Multimodal ingestion → Weaviate → rerank → LLM via OnDemand API |
| `RAG_2.py` | Leaner rewrite: Weaviate schema setup, chunk ingestion, **hybrid retrieval**, cross-encoder rerank |
| `not_main.py` | Minimal FastAPI skeleton (`/`, `/health`, `/items` CRUD), the starting point for an API |
| `scripts/start_fastapi.sh` | uvicorn launcher; loads `.env`, env vars `APP_MODULE`, `HOST`, `PORT`, `RELOAD` |

### Pipeline (`Python_code_1.py`)

1. **Ingest** (`user_sends_file`)
   - **PDF:** text via PyMuPDF; pages with no text layer are detected and **OCR'd with Tesseract**.
   - **Audio** (lectures): transcribed with **faster-whisper**, keeping `[mm:ss]` timestamps.
2. **Chunk:** LangChain `SemanticChunker` over `all-MiniLM-L6-v2` embeddings. Chunks keep
   source name and, for audio, `start_time` / `reference_end_time`, so answers can cite the
   moment in the lecture.
3. **Store:** embedded Weaviate (`EmbeddedOptions`), collection `DtuDoubt`.
4. **Retrieve:** top-12 `near_vector`, then **cross-encoder rerank** to the top 5
   (`cross-encoder/ms-marco-MiniLM-L-6-v2`).
5. **Generate:** prompt with retrieved context and formatted chat history sent to the
   **OnDemand chat API**; the response is streamed back character by character.

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # also needs the tesseract binary: brew/apt install tesseract
cp .env.example .env                     # set ONDEMAND_API_KEY
python Python_code_1.py                  # pipeline
APP_MODULE=not_main:app ./scripts/start_fastapi.sh   # API skeleton on :8000
```

## Known issues

- `start_fastapi.sh` defaults to `main:app`, but there is no `main.py`; pass
  `APP_MODULE=not_main:app`.
- The API skeleton isn't wired to the RAG pipeline yet (v2 does that).
- Embedded Weaviate stores data in a local temp directory; it's not persisted between machines.
