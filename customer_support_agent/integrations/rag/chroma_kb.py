from __future__ import annotations
import hashlib
from pathlib import Path
from typing import Any

import chromadb
from chromadb.types import Metadata
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter

from customer_support_agent.core.settings import Settings


class KnowledgeBaseService:

    def __init__(self, settings: Settings):
        self._settings = settings

        self._client = chromadb.PersistentClient(
            path=str(settings.chroma_rag_path)
        )

        # Keep OpenAI embeddings separate from a previous provider collection.
        self._collection_name = "support_kb_openai"
        self._embedding_function = self._build_embedding_function()

        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            embedding_function=self._embedding_function,
        )

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.rag_chunk_size,
            chunk_overlap=settings.rag_chunk_overlap,
        )

    def _build_embedding_function(self) -> Any:
        if not self._settings.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is required for OpenAI embeddings. Add it to the .env file."
            )

        try:
            return embedding_functions.OpenAIEmbeddingFunction(
                api_key=self._settings.openai_api_key,
                model_name=self._settings.openai_embedding_model,
            )
        except Exception as exc:
            raise RuntimeError(
                "OpenAI embedding initialization failed. Install `openai` "
                "and verify OPENAI_API_KEY."
            ) from exc
    

    def ingest_directory(
    self,
    directory: Path,
    clear_existing: bool = False,
    ) -> dict[str, int]:
        source_files = sorted(
            [
                *directory.glob("**/*.md"),
                *directory.glob("**/*.txt"),
            ]
        )

        docs: list[str] = []
        ids: list[str] = []
        metadatas: list[Metadata] = []

        for file_path in source_files:
            text = file_path.read_text(encoding="utf-8")
            chunks = self._splitter.split_text(text)

            for index, chunk in enumerate(chunks):
                chunk_hash = hashlib.sha1(
                    chunk.encode("utf-8")
                ).hexdigest()[:10]

                doc_id = f"{file_path.stem}-{index}-{chunk_hash}"

                docs.append(chunk)
                ids.append(doc_id)
                metadatas.append(
                    {
                        "source": file_path.name,
                        "chunk_index": index,
                    }
                )
        
        # Chroma requires one vector for every ID, document, and metadata
        # entry. Generate the batch before replacing an existing collection.
        embeddings = self._embed_documents(docs) if docs else []

        # Do not discard a working collection until all replacement embeddings
        # have been generated successfully.
        if clear_existing:
            self._client.delete_collection(name=self._collection_name)
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                embedding_function=self._embedding_function,
            )

        if docs:
            self._collection.upsert(
                documents=docs,
                embeddings=embeddings,
                ids=ids,
                metadatas=metadatas,
            )

        return {
            "files_indexed":len(source_files),
            "chunks_indexed":len(docs),
            "collection_count":self._collection.count()
        }

    def _embed_documents(self, documents: list[str]) -> list[Any]:
        embeddings = self._embedding_function(documents)

        if len(embeddings) != len(documents):
            raise RuntimeError("Embedding count does not match document count.")
        return embeddings


    def search(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        collection_count = self._collection.count()
        if collection_count == 0:
            return []

        # Chroma rejects a query asking for more results than the collection
        # contains.  A newly ingested knowledge base often has fewer chunks
        # than the configured default, so cap the request before querying.
        result_limit = min(top_k or self._settings.rag_top_k, collection_count)

        results = self._collection.query(
            query_texts=[query],
            n_results=result_limit,
            include=["documents", "metadatas", "distances"],
        )

        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]

        combined: list[dict[str, Any]] = []

        for i, document in enumerate(documents):
            metadata = metadatas[i] if i < len(metadatas) else {}
            distance = distances[i] if i < len(distances) else None

            combined.append(
                {
                    "content": document,
                    "source": metadata.get("source", "unknown"),
                    "distance": distance,
                }
            )
        return combined
