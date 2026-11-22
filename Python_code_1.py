import os
import sys
import time
import json 
import logging
import datetime
import requests
import re
import io
import shutil  
from pathlib import Path 
from dataclasses import dataclass
from dotenv import load_dotenv
from PIL import Image
import pytesseract
import pymupdf
import weaviate
from weaviate.embedded import EmbeddedOptions
from sentence_transformers import CrossEncoder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from faster_whisper import WhisperModel
import fastapi
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from fastapi.responses import StreamingResponse



app = FastAPI(title="DTUnator RAG API")

load_dotenv()
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

ONDEMAND_API_KEY = os.getenv("ONDEMAND_API_KEY")
if not ONDEMAND_API_KEY:
    logger.warning("ONDEMAND_API_KEY missing")

DB_PATH = "./vector_db"
INDEX_NAME = "DtuDoubt"
TEMP_UPLOAD_DIR = "./temp_uploads"

ONDEMAND_URL = "https://api.on-demand.io/chat/v1/sessions/query"
ONDEMAND_ENDPOINT_ID = "predefined-openai-gpt4.1-nano"

MAX_HISTORY_TURNS = 4

WHISPER_MODEL_SIZE = "distil-medium.en"


@dataclass(frozen=True)
class AppConfig:
    PAGE_TITLE: str = "DTUnator"
    PAGE_ICON: str = "⚡"
    RATE_LIMIT_SEC: float = 1.0


os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)


def load_models():
    logger.info("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    logger.info("Loading cross-encoder...")
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    logger.info("Loading semantic chunker...")
    text_splitter = SemanticChunker(embeddings)

    return embeddings, cross_encoder, text_splitter


embeddings_model, cross_encoder_model, text_splitter_model = load_models()


def get_weaviate_client():
    client = weaviate.WeaviateClient(
        embedded_options=EmbeddedOptions(persistence_data_path=DB_PATH)
    )
    client.connect()
    return client


def clean_text(text):
    text = re.sub(r"\d+/\d+", "", text)
    text = re.sub(r"\d{1,4}[-/]\d{1,2}[-/]\d{1,4}", "", text)
    text = re.sub(r"(\[\d+:\d+\])", r"\n\1", text)

    replacements = {"": "t", "": "t", "": "ti", "": "f", "": "ti"}
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def create_chunks(text, source_name, is_general=False, is_timed_source=False):
    documents = text_splitter_model.create_documents([text])
    chunk_data = []
    ts_pattern = r"\[(\d+:\d+)\]"

    for i, doc in enumerate(documents):
        content = doc.page_content
        start_time = None
        last_timestamp = None
        if is_timed_source:
            matches = list(re.finditer(ts_pattern, content))
            if matches:
                start_time = matches[0].group(1)
                last_timestamp = matches[-1].group(1)
            content = re.sub(ts_pattern, " ", content)
            content = " ".join(content.split())

        chunk_obj = {
            "page_content": content,
            "metadata": {
                "source": source_name,
                "chunk_index": i + 1,
                "is_general": is_general,
                "type": "general_knowledge",
            },
        }

        if start_time:
            chunk_obj["metadata"]["start_time"] = start_time
        if last_timestamp and last_timestamp != start_time:
            chunk_obj["metadata"]["reference_end_time"] = last_timestamp

        chunk_data.append(chunk_obj)

    return chunk_data


def is_scanned_page(page):
    text = page.get_text().strip()
    return len(text) < 50


def process_scanned_page(page):
    pix = page.get_pixmap(dpi=300)
    img_data = pix.tobytes("png")
    image = Image.open(io.BytesIO(img_data))
    return pytesseract.image_to_string(image, lang="eng")


def extract_from_audio(audio_path):
    try:
        model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
        segments, info = model.transcribe(audio_path, beam_size=1, vad_filter=True)

        full_text = ""
        for segment in segments:
            start = int(segment.start)
            timestamp = f"[{start // 60}:{start % 60:02d}]"
            full_text += f"{timestamp} {segment.text}\n"
        return full_text

    except Exception as e:
        logger.error(f"Audio Extraction Error: {e}")
        return ""


def user_sends_file(file_path):
    """
    Master function to handle file ingestion for RAG.
    Returns: List of chunk dictionaries.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        return {"error": "File Not Found"}

    ext = file_path.suffix.lower()
    chunks = []

    try:
        if ext == ".pdf":
            doc = pymupdf.open(file_path)
            all_text = ""
            for page in doc:
                if is_scanned_page(page):
                    all_text += process_scanned_page(page) + "\n"
                else:
                    page_text = page.get_text("text")
                    if isinstance(page_text, str):
                        all_text += page_text + "\n"
                    elif isinstance(page_text, list):
                        all_text += "\n".join(str(x) for x in page_text) + "\n"
                    elif isinstance(page_text, dict):

                        text_val = page_text.get("text") or page_text.get("content") or ""
                        if text_val:
                            all_text += str(text_val) + "\n"
                        else:
                            all_text += json.dumps(page_text) + "\n"
                    else:
                        all_text += str(page_text) + "\n"

            doc.close()

            cleaned_text = clean_text(all_text)
            chunks = create_chunks(cleaned_text, file_path.name)

        elif ext in [".mp3", ".wav", ".aac", ".flac", ".ogg", ".wma", ".m4a"]:
            raw_text = extract_from_audio(str(file_path))
            if not raw_text or "Error" in raw_text:
                return {"error": f"Audio extraction failed: {raw_text}"}

            cleaned_text = clean_text(raw_text)
            chunks = create_chunks(cleaned_text, file_path.name, is_timed_source=True)

        elif ext in [".txt", ".md"]:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            cleaned_text = clean_text(content)
            chunks = create_chunks(cleaned_text, file_path.name, is_timed_source=False)

        else:
            return {"error": f"Unsupported file type: {ext}"}

        return chunks

    except Exception as e:
        logger.error(f"Error processing file: {e}")
        return {"error": str(e)}


def retrieve_docs_from_weaviate(query, k=12):
    """Standard retrieval from Weaviate DB"""
    client = get_weaviate_client()
    docs = []
    try:
        query_vector = embeddings_model.embed_query(query)
        result = client.collections.get(INDEX_NAME).query.near_vector(
            near_vector=query_vector, limit=k, return_properties=["text", "source"]
        )

        for obj in result.objects:
            text = obj.properties.get("text")
            if isinstance(text, str) and text.strip():
                docs.append(
                    {
                        "content": text,
                        "metadata": {"source": obj.properties.get("source", "Unknown")},
                    }
                )
    except Exception as e:
        logger.error(f"Retrieval error: {e}")
    finally:
        client.close()

    return docs


def cross_rerank(query, docs, top_k=5):
    if not docs:
        return []

    pairs = []
    valid_docs = []

    for d in docs:
        text = d.get("content") or d.get("page_content")
        if text:
            pairs.append((query, text))
            valid_docs.append(d)

    if not pairs:
        return []

    scores = cross_encoder_model.predict(pairs)

    ranked = sorted(zip(valid_docs, scores), key=lambda x: x[1], reverse=True)
    # 
    results = []
    for i, (doc, score) in enumerate(ranked[:top_k]):
        content = doc.get("content") or doc.get("page_content")
        results.append(
            {
                "rank": i + 1,
                "score": float(score),
                "content": content,
                "metadata": doc["metadata"],
            }
        )

    return results


def format_history(messages):
    history = messages[-MAX_HISTORY_TURNS * 2 :]
    lines = []
    for msg in history:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


def call_llm_api(prompt, system_instruction):
    payload = {
        "endpointId": ONDEMAND_ENDPOINT_ID,
        "responseMode": "sync",
        "query": f"{system_instruction}\n\n{prompt}",
    }

    headers = {"apikey": ONDEMAND_API_KEY, "Content-Type": "application/json"}

    try:
        response = requests.post(
            ONDEMAND_URL, json=payload, headers=headers, timeout=60
        )
        response.raise_for_status()
        return response.json()["data"]["answer"]
    except Exception as e:
        logger.error(e)
        return "Sorry, something went wrong while answering."


def generate_rag_response(user_query, history, uploaded_chunks=None):
    """
    Decides whether to use the Uploaded File chunks OR the Weaviate DB.
    """

    if uploaded_chunks:
        reranked = cross_rerank(user_query, uploaded_chunks, top_k=7)
        history_block = format_history(history)
        context_text = "\n\n".join(d["content"] for d in reranked)

        prompt = f"""
Conversation so far:
{history_block}

Uploaded Document Context:
{context_text}

Question:
{user_query}
"""
        system_instruction = (
            "You are DTUnator analysis mode.\n"
            "The user has uploaded a specific document.\n"
            "Answer the question ONLY using the provided as well as uploaded Document Context.\n"
            "Do not use outside knowledge or the vector database.\n"
            "\n"
            "Be accurate and direct."
        )

#  ?? ANswers even if not in the documnets 

        return call_llm_api(prompt, system_instruction)

   
    else:
        retrieved = retrieve_docs_from_weaviate(user_query)
        reranked = cross_rerank(user_query, retrieved)
        history_block = format_history(history)

        strong_context = bool(reranked) and reranked[0]["score"] > 6.5

        if strong_context:
            context_text = "\n\n".join(d["content"] for d in reranked)
            prompt = f"Conversation:\n{history_block}\n\nNotes:\n{context_text}\n\nQ:\n{user_query}"

            system_instruction = (
                "You are DTUnator, a helpful DTU study assistant.\n"
                "Use the lecture notes as your primary source.\n"
                "You are intellectually cursious and humble"
                "Have a bubbly personality"
                "Fix the font, write the reponse in a next line after the user query"
                "Ask permision to search on prior knowledge if neccessary.\n"
                "Be accurate. If unsure, say so.\n"
                "Do NOT fabricate facts.\n"
                "Ask before using Prior Information\n"
            )

        else:
            prompt = f"Conversation:\n{history_block}\n\nUser Input:\n{user_query}"
            system_instruction = (
                "You are DTUnator, a helpful DTU study assistant.\n"
                "You do not have access to specific lecture notes right now.\n"
                "Refuse to answer questions that require specific lecture knowledge.\n"
            )

        return call_llm_api(prompt, system_instruction)


class LLMService:
    def generate_response(self, question: str, history: list, uploaded_chunks=None):
        # Pass the uploaded_chunks to the generator logic
        response = generate_rag_response(question, history, uploaded_chunks)

        for word in response.split():
            yield word + " "
            time.sleep(0.015)


def ensure_schema(client):
    """Ensure Weaviate schema exists."""
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