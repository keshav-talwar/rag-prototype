import os
import re
import logging
from typing import List, Dict
from dotenv import load_dotenv

import weaviate
from weaviate.embedded import EmbeddedOptions

from sentence_transformers import CrossEncoder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_PATH = "./vector_db"
INDEX_NAME = "RAGDocs"

EMBED_MODEL = "all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

MIN_CE_SCORE = 6.5
TOP_K_RETRIEVE = 20
TOP_K_RERANK = 5

def load_models():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    return embeddings, cross_encoder, splitter


embeddings_model, cross_encoder_model, text_splitter = load_models()

def get_weaviate_client():
    client = weaviate.WeaviateClient(
        embedded_options=EmbeddedOptions(
            persistence_data_path=DB_PATH
        )
    )
    client.connect()
    return client
def ensure_schema(client):
    if client.collections.exists(INDEX_NAME):
        return

    client.collections.create(
        name=INDEX_NAME,
        properties=[
            {"name": "text", "dataType": ["text"]},
            {"name": "source", "dataType": ["text"]},
        ],
        vectorizer_config=None
    )
def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    return text.strip()
def ingest_chunks(chunks: List[Dict]):
    client = get_weaviate_client()
    ensure_schema(client)
    col = client.collections.get(INDEX_NAME)

    with col.batch.dynamic() as batch:
        for c in chunks:
            vec = embeddings_model.embed_query(c["content"])
            batch.add_object(
                properties={
                    "text": c["content"],
                    "source": c["metadata"]["source"],
                },
                vector=vec
            )

    client.close()
def hybrid_retrieve(query: str, k: int) -> List[Dict]:
    client = get_weaviate_client()
    col = client.collections.get(INDEX_NAME)

    vector = embeddings_model.embed_query(query)

    results = col.query.hybrid(
        query=query,
        vector=vector,
        alpha=0.5,  # balance BM25 vs vector
        limit=k,
        return_properties=["text", "source"]
    )

    docs = []
    for o in results.objects:
        docs.append({
            "content": o.properties["text"],
            "metadata": {"source": o.properties.get("source", "unknown")}
        })

    client.close()
    return docs
def cross_rerank(query: str, docs: List[Dict]) -> List[Dict]:
    if not docs:
        return []

    pairs = [(query, d["content"]) for d in docs]
    scores = cross_encoder_model.predict(pairs)

    ranked = sorted(
        zip(docs, scores),
        key=lambda x: x[1],
        reverse=True
    )

    results = []
    for i, (doc, score) in enumerate(ranked[:TOP_K_RERANK]):
        results.append({
            "rank": i + 1,
            "score": float(score),
            "content": doc["content"],
            "metadata": doc["metadata"]
        })

    return results
