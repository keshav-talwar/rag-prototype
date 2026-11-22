# RAG Prototype

Early retrieval-augmented generation prototype (Jan–Feb 2026), before
[dtunator-rag-sft-rl](https://github.com/Keshav-cse/dtunator-rag-sft-rl).

- `Python_code_1.py`: document ingestion (PDF via PyMuPDF, OCR via Tesseract, audio via
  faster-whisper) and querying through the OnDemand API
- `RAG_2.py`: embedded Weaviate vector store + LangChain semantic chunking + HuggingFace embeddings
- `not_main.py`: minimal FastAPI app
- `scripts/start_fastapi.sh`: uvicorn launcher

```bash
cp .env.example .env    # add your ONDEMAND_API_KEY
pip install -r requirements.txt
./scripts/start_fastapi.sh
```
