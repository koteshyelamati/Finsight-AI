from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def load_text_files(directory: Path) -> list[dict[str, str]]:
    docs = []
    for fp in directory.glob("*.txt"):
        content = fp.read_text(encoding="utf-8")
        docs.append({"source": fp.name, "content": content})
        logger.info("Loaded %s (%d chars)", fp.name, len(content))
    return docs


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks


def ingest(source_dir: Path, collection_name: str = "finsight") -> None:
    try:
        import chromadb
    except ImportError:
        raise SystemExit("chromadb is required: pip install chromadb")

    docs = load_text_files(source_dir)
    if not docs:
        logger.warning("No .txt files found in %s", source_dir)
        return

    client = chromadb.PersistentClient(path=str(Path(__file__).parent / "chroma_db"))
    collection = client.get_or_create_collection(collection_name)

    ids, texts, metadatas = [], [], []
    for doc in docs:
        chunks = chunk_text(doc["content"])
        for i, chunk in enumerate(chunks):
            ids.append(f"{doc['source']}::{i}")
            texts.append(chunk)
            metadatas.append({"source": doc["source"], "chunk": i})

    collection.upsert(ids=ids, documents=texts, metadatas=metadatas)
    logger.info("Upserted %d chunks into collection '%s'", len(ids), collection_name)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Ingest financial documents into ChromaDB")
    parser.add_argument("--source", default=str(Path(__file__).parent / "sample_filings"))
    parser.add_argument("--collection", default="finsight")
    args = parser.parse_args()
    ingest(Path(args.source), args.collection)
