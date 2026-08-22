"""Central configuration, loaded from environment variables (.env)."""
import os

from dotenv import load_dotenv

load_dotenv()  # reads the .env file if present


class Config:
    # Vector DB.
    # The default matches this repository's docker-compose, which publishes Qdrant on
    # host port 6533 (not the usual 6333) so it cannot collide with another local Qdrant.
    # The previous default of 6333 pointed at a port nothing in this project listens on.
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6533")
    QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "nfip_sfip")

    # Embeddings
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

    # Generation
    LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama")  # "ollama" or "hf"
    OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    HF_BASE_MODEL = os.getenv("HF_BASE_MODEL", "microsoft/Phi-3-mini-4k-instruct")
    LORA_ADAPTER_PATH = os.getenv("LORA_ADAPTER_PATH", "./finetune/adapter")

    # Corpus. "nfip" is the real, licensed corpus under data/corpus/; "sample" is the
    # two synthetic policy files in data/, kept for offline demos and fixtures.
    CORPUS = os.getenv("CORPUS", "nfip")

    # Retrieval
    TOP_K = int(os.getenv("TOP_K", "5"))
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))     # characters (frozen; see eval/retrieval_config.json)
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))

    # BGE models are trained with an asymmetric retrieval instruction: the QUERY carries a
    # prefix and the passages do not. BAAI documents this for bge-*-en-v1.5 and reports it
    # matters most for short queries, which is exactly this workload. Omitting it - as this
    # project did - embeds questions and passages in mismatched styles.
    # Set BGE_QUERY_PREFIX="" to disable, e.g. when using a non-BGE embedder.
    BGE_QUERY_PREFIX = os.getenv(
        "BGE_QUERY_PREFIX",
        "Represent this sentence for searching relevant passages: ",
    )


cfg = Config()
