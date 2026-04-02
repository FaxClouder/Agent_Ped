from __future__ import annotations

import json
import math
import re
from hashlib import sha256
from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.docstore.document import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from evac_agent.config import get_embedding_kwargs, get_settings


SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf"}
MANIFEST_FILE = "manifest.json"
_RRF_K = 60
_RUNTIME_CACHE: dict[str, tuple[FAISS, list[Document]]] = {}
_TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]|\w+")


class LocalHashEmbeddings(Embeddings):
    def __init__(self, dimensions: int = 1536):
        self.dimensions = dimensions

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = _TOKEN_PATTERN.findall((text or "").lower())
        if not tokens:
            return vector

        for token in tokens:
            digest = sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if (digest[4] & 1) else -1.0
            weight = 1.0 + (digest[5] / 255.0) * 0.1
            vector[idx] += sign * weight

        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0:
            return vector
        return [v / norm for v in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def _iter_knowledge_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def _build_manifest(paths: list[Path]) -> dict[str, object]:
    files = []
    for path in paths:
        stat = path.stat()
        files.append(
            {
                "path": path.as_posix(),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )

    digest = sha256(json.dumps(files, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    return {"fingerprint": digest, "files": files}


def _load_text_document(path: Path) -> list[Document]:
    text = path.read_text(encoding="utf-8")
    return [
        Document(
            page_content=text,
            metadata={"source": str(path), "doc_type": path.suffix.lower().lstrip(".")},
        )
    ]


def _load_pdf_document(path: Path) -> list[Document]:
    pages = PyPDFLoader(str(path)).load()
    for index, page in enumerate(pages, start=1):
        page.metadata["source"] = str(path)
        page.metadata["page"] = index
        page.metadata["doc_type"] = "pdf"
    return pages


def _load_documents(root: Path) -> list[Document]:
    docs: list[Document] = []
    for path in _iter_knowledge_files(root):
        if path.suffix.lower() == ".pdf":
            docs.extend(_load_pdf_document(path))
        else:
            docs.extend(_load_text_document(path))
    return docs


def _chunk_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=120,
        separators=["\n## ", "\n### ", "\n", "。", "；", "，", " "],
    )
    return splitter.split_documents(docs)


def _manifest_changed(store_path: Path, current_manifest: dict[str, object]) -> bool:
    manifest_path = store_path / MANIFEST_FILE
    if not manifest_path.exists():
        return True

    saved_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return saved_manifest.get("fingerprint") != current_manifest.get("fingerprint")


def _cache_key(knowledge_root: Path, manifest_fingerprint: str) -> str:
    return f"{knowledge_root.resolve()}::{manifest_fingerprint}"


def _build_embeddings() -> tuple[Embeddings, str]:
    settings = get_settings()
    has_embedding_key = bool((settings.embedding_api_key or "").strip())
    if has_embedding_key:
        embeddings = OpenAIEmbeddings(**get_embedding_kwargs(settings))
        signature = f"openai_compatible::{settings.embedding_model}::{settings.embedding_base_url or 'default'}"
        return embeddings, signature

    embeddings = LocalHashEmbeddings()
    return embeddings, "local_hash::1536"


def _extract_index_documents(vectorstore: FAISS) -> list[Document]:
    docstore = getattr(vectorstore, "docstore", None)
    store_dict = getattr(docstore, "_dict", {})
    docs = [doc for doc in store_dict.values() if isinstance(doc, Document)]
    return docs


def build_or_load_vectorstore(force_rebuild: bool = False) -> tuple[FAISS, list[Document], Embeddings]:
    settings = get_settings()
    embeddings, embedding_signature = _build_embeddings()
    store_path = settings.vector_store_dir
    knowledge_root = settings.knowledge_dir
    knowledge_files = _iter_knowledge_files(knowledge_root)

    if not knowledge_files:
        raise FileNotFoundError(f"No knowledge files found under: {knowledge_root}")

    manifest = _build_manifest(knowledge_files)
    fingerprint = str(manifest["fingerprint"])
    key = _cache_key(knowledge_root, f"{fingerprint}::{embedding_signature}")

    if not force_rebuild and key in _RUNTIME_CACHE:
        vectorstore, chunks = _RUNTIME_CACHE[key]
        return vectorstore, chunks, embeddings

    index_exists = (store_path / "index.faiss").exists()
    if index_exists and not force_rebuild and not _manifest_changed(store_path, manifest):
        vectorstore = FAISS.load_local(
            str(store_path),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        chunks = _extract_index_documents(vectorstore)
        if not chunks:
            chunks = _chunk_documents(_load_documents(knowledge_root))
        _RUNTIME_CACHE[key] = (vectorstore, chunks)
        return vectorstore, chunks, embeddings

    docs = _load_documents(knowledge_root)
    chunks = _chunk_documents(docs)
    store_path.mkdir(parents=True, exist_ok=True)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(str(store_path))
    (store_path / MANIFEST_FILE).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _RUNTIME_CACHE.clear()
    _RUNTIME_CACHE[key] = (vectorstore, chunks)
    return vectorstore, chunks, embeddings


def _doc_key(doc: Document) -> str:
    source = doc.metadata.get("source", "unknown")
    page = doc.metadata.get("page", "na")
    head = doc.page_content[:160]
    return sha256(f"{source}|{page}|{head}".encode("utf-8")).hexdigest()


def _source_label(doc: Document) -> str:
    source = doc.metadata.get("source", "unknown")
    page = doc.metadata.get("page")
    return f"{source}#page={page}" if page else source


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return numerator / (norm_a * norm_b)


def _rrf_merge(
    queries: list[str],
    dense_retriever,
    sparse_retriever,
    dense_weight: float,
    sparse_weight: float,
) -> list[tuple[Document, float]]:
    scores: dict[str, float] = {}
    docs_by_key: dict[str, Document] = {}

    for query in queries:
        dense_docs = dense_retriever.invoke(query)
        sparse_docs = sparse_retriever.invoke(query)

        for rank, doc in enumerate(dense_docs):
            key = _doc_key(doc)
            docs_by_key[key] = doc
            scores[key] = scores.get(key, 0.0) + dense_weight / (_RRF_K + rank + 1)

        for rank, doc in enumerate(sparse_docs):
            key = _doc_key(doc)
            docs_by_key[key] = doc
            scores[key] = scores.get(key, 0.0) + sparse_weight / (_RRF_K + rank + 1)

    ranked = sorted(((docs_by_key[key], score) for key, score in scores.items()), key=lambda item: item[1], reverse=True)
    return ranked


def retrieve_context(
    question: str,
    queries: list[str],
    top_k: int = 6,
    force_rebuild: bool = False,
) -> tuple[str, list[str]]:
    vectorstore, chunk_docs, embeddings = build_or_load_vectorstore(force_rebuild=force_rebuild)
    dense_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": max(top_k + 2, 8), "fetch_k": max(top_k * 4, 20)},
    )
    sparse_retriever = BM25Retriever.from_documents(chunk_docs)
    sparse_retriever.k = max(top_k + 2, 8)

    merged = _rrf_merge(
        queries=queries,
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        dense_weight=0.6,
        sparse_weight=0.4,
    )
    if not merged:
        return "", []

    query_vector = embeddings.embed_query(f"{question}\n" + "\n".join(queries[:4]))

    candidate_docs = [doc for doc, _ in merged[: max(top_k * 4, 16)]]
    candidate_vectors = embeddings.embed_documents([doc.page_content for doc in candidate_docs])

    reranked: list[tuple[Document, float]] = []
    for (doc, fusion_score), doc_vector in zip(merged[: len(candidate_docs)], candidate_vectors):
        semantic_score = _cosine_similarity(query_vector, doc_vector)
        final_score = fusion_score * 0.35 + semantic_score * 0.65
        reranked.append((doc, final_score))

    top_docs = [doc for doc, _ in sorted(reranked, key=lambda item: item[1], reverse=True)[:top_k]]
    merged_context = "\n\n---\n\n".join(doc.page_content for doc in top_docs)
    sources = [_source_label(doc) for doc in top_docs]
    return merged_context, sources
