"""Central configuration, loaded from environment variables (.env)."""
import os

from dotenv import load_dotenv

load_dotenv()  # reads the .env file if present


class Config:
    # Vector DB
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "insurance_docs")

    # Embeddings
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

    # Generation
    LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama")  # "ollama" or "hf"
    OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    HF_BASE_MODEL = os.getenv("HF_BASE_MODEL", "microsoft/Phi-3-mini-4k-instruct")
    LORA_ADAPTER_PATH = os.getenv("LORA_ADAPTER_PATH", "./finetune/adapter")

    # Retrieval
    TOP_K = int(os.getenv("TOP_K", "4"))
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))     # characters
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))


cfg = Config()
