"""从指定 PDF 页码构建并检索中医日常养生知识库。"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader, PdfWriter
from dotenv import load_dotenv


RAG_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = RAG_ROOT.parents[1]
load_dotenv(PROJECT_ROOT / ".env")
SOURCE_PDF = RAG_ROOT / "了不起的中医养生妙招（全二册）.pdf"
SOURCE_TITLE = "了不起的中医养生妙招（全二册）"
PAGE_START = 8
PAGE_END = 126
MINERU_MAX_PAGES_PER_REQUEST = 20
DEFAULT_COLLECTION = "tcm_wellness_pdf_v1"
DEFAULT_PERSIST_DIRECTORY = RAG_ROOT / "chroma" / "tcm_wellness"
SUBSET_PDF = RAG_ROOT / ".cache" / "tcm_wellness_pages_8_126.pdf"
logger = logging.getLogger("fitness_assistant")


class TCMRAGNotReadyError(RuntimeError):
    """知识库还未构建或持久化目录不可用。"""


@dataclass(frozen=True)
class RetrievedTCMChunk:
    source_id: str
    title: str
    content: str
    page_start: int | None
    page_end: int | None


@dataclass(frozen=True)
class TCMRetrievalTrace:
    """一次混合检索的可观测信息，不包含用户健康原始内容。"""

    original_query: str
    rewritten_query: str
    vector_candidates: int
    bm25_candidates: int
    fused_candidates: int
    rerank_applied: bool
    elapsed_ms: int


@dataclass(frozen=True)
class TCMRetrievalResult:
    chunks: list[RetrievedTCMChunk]
    trace: TCMRetrievalTrace


def _persist_directory() -> Path:
    configured = os.getenv("TCM_RAG_PERSIST_DIRECTORY", "").strip()
    return Path(configured).expanduser() if configured else DEFAULT_PERSIST_DIRECTORY


def _collection_name() -> str:
    return os.getenv("TCM_RAG_COLLECTION", DEFAULT_COLLECTION).strip() or DEFAULT_COLLECTION


def _embedding_model() -> str:
    return os.getenv("TCM_RAG_EMBEDDING_MODEL", "text-embedding-v1").strip() or "text-embedding-v1"


def _positive_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except ValueError:
        return default


def _rerank_enabled() -> bool:
    return os.getenv("TCM_RAG_ENABLE_RERANK", "true").strip().lower() not in {"0", "false", "no"}


def _create_page_subset(source: Path = SOURCE_PDF, target: Path = SUBSET_PDF) -> Path:
    """生成仅含物理第 8-126 页的临时 PDF，排除封面、版权和附录杂页。"""
    if not source.is_file():
        raise FileNotFoundError(f"未找到中医 RAG 源文件：{source}")
    reader = PdfReader(str(source))
    if len(reader.pages) < PAGE_END:
        raise ValueError(f"PDF 只有 {len(reader.pages)} 页，无法读取到第 {PAGE_END} 页")

    source_fingerprint = f"{source.stat().st_mtime_ns}:{source.stat().st_size}:{PAGE_START}:{PAGE_END}"
    fingerprint_path = target.with_suffix(".fingerprint")
    if target.is_file() and fingerprint_path.is_file() and fingerprint_path.read_text() == source_fingerprint:
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    writer = PdfWriter()
    for page_index in range(PAGE_START - 1, PAGE_END):
        writer.add_page(reader.pages[page_index])
    with target.open("wb") as output_file:
        writer.write(output_file)
    fingerprint_path.write_text(source_fingerprint)
    return target


def _create_embeddings() -> Any:
    try:
        from langchain_community.embeddings.dashscope import DashScopeEmbeddings
    except ImportError as exc:
        raise RuntimeError("缺少 DashScope 向量化依赖，请安装 requirements.txt") from exc
    if not os.getenv("DASHSCOPE_API_KEY", "").strip():
        raise RuntimeError("缺少 DASHSCOPE_API_KEY，无法进行 DashScope 向量化")
    return DashScopeEmbeddings(model=_embedding_model())


def _create_vector_store() -> Any:
    try:
        from langchain_chroma import Chroma
    except ImportError as exc:
        raise RuntimeError("缺少 Chroma 依赖，请安装 requirements.txt") from exc
    return Chroma(
        collection_name=_collection_name(),
        persist_directory=str(_persist_directory()),
        embedding_function=_create_embeddings(),
    )


def load_tcm_wellness_documents() -> list[Document]:
    """用 MinerULoader 加载裁切后的 PDF；所有内容都来自原 PDF 第 8-126 页。"""
    try:
        from langchain_mineru import MinerULoader
    except ImportError as exc:
        raise RuntimeError("缺少 MinerU Loader 依赖，请安装 requirements.txt") from exc

    subset = _create_page_subset()
    # MinerU Flash API 每次最多处理 20 页。对裁切 PDF 做 20 页一批的远程解析，
    # 既避免 119 个单页任务，也不让杂页进入任何一个批次。
    documents: list[Document] = []
    total_pages = PAGE_END - PAGE_START + 1
    for batch_start in range(1, total_pages + 1, MINERU_MAX_PAGES_PER_REQUEST):
        batch_end = min(batch_start + MINERU_MAX_PAGES_PER_REQUEST - 1, total_pages)
        batch_documents = MinerULoader(
            source=str(subset), pages=f"{batch_start}-{batch_end}"
        ).load()
        for document in batch_documents:
            document.metadata.update(
                {
                    "source_page_start": PAGE_START + batch_start - 1,
                    "source_page_end": PAGE_START + batch_end - 1,
                    "mineru_page_range": f"{batch_start}-{batch_end}",
                }
            )
        documents.extend(batch_documents)
    if not documents:
        raise RuntimeError("MinerU 未从第 8-126 页提取到可索引文本")
    for index, document in enumerate(documents):
        document.metadata.update(
            {
                "source": str(SOURCE_PDF),
                "source_title": SOURCE_TITLE,
                "mineru_document_index": index,
            }
        )
    return documents


def split_tcm_wellness_documents(documents: list[Document]) -> list[Document]:
    """按 Markdown/中文自然边界切分，保留小节语义并兼顾召回粒度。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=120,
        separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", "。", "！", "？", "；", "，", ""],
        length_function=len,
        is_separator_regex=False,
    )
    chunks = splitter.split_documents(documents)
    for index, chunk in enumerate(chunks):
        normalized = " ".join(chunk.page_content.split())
        digest = hashlib.sha256(
            f"{SOURCE_PDF.name}:{PAGE_START}:{PAGE_END}:{index}:{normalized}".encode("utf-8")
        ).hexdigest()[:20]
        chunk.metadata.update(
            {
                "source_id": f"tcm-pdf:{digest}",
                "chunk_index": index,
                "source_title": SOURCE_TITLE,
                # 每个块都明确标记其所属的已审核文档范围；精确页码可在
                # 后续接入 MinerU 页级定位时再增强，不影响本次范围隔离。
                "source_page_start": PAGE_START,
                "source_page_end": PAGE_END,
            }
        )
    return chunks


def build_tcm_wellness_index(*, reset: bool = False) -> dict[str, Any]:
    """加载、切分、向量化并持久化 Chroma 知识库。"""
    documents = load_tcm_wellness_documents()
    chunks = split_tcm_wellness_documents(documents)
    if not chunks:
        raise RuntimeError("切分后没有可写入 Chroma 的文本块")

    vector_store = _create_vector_store()
    if reset:
        try:
            vector_store.delete_collection()
        except Exception:
            pass  # 第一次构建时集合尚不存在。
        vector_store = _create_vector_store()

    vector_store.add_documents(chunks, ids=[chunk.metadata["source_id"] for chunk in chunks])
    manifest = {
        "source": str(SOURCE_PDF),
        "source_page_start": PAGE_START,
        "source_page_end": PAGE_END,
        "document_count": len(documents),
        "chunk_count": len(chunks),
        "collection": _collection_name(),
        "embedding_model": _embedding_model(),
    }
    persist_directory = _persist_directory()
    persist_directory.mkdir(parents=True, exist_ok=True)
    (persist_directory / "build_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    get_tcm_wellness_retriever.cache_clear()
    _get_bm25_index.cache_clear()
    return manifest


@lru_cache(maxsize=1)
def get_tcm_wellness_retriever() -> Any:
    """获取 MMR 检索器，避免相邻重复段落挤占中医 Agent 上下文。"""
    manifest = _persist_directory() / "build_manifest.json"
    if not manifest.is_file():
        raise TCMRAGNotReadyError(
            "中医 PDF 知识库尚未构建；请在 backend 目录执行 python -m rag.build_tcm_wellness_index"
        )
    return _create_vector_store().as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 12, "lambda_mult": 0.65},
    )


def _tokenize_for_bm25(text: str) -> list[str]:
    """无额外分词模型的中文 BM25 分词：保留词串与相邻汉字二元组。"""
    normalized = re.sub(r"\s+", "", text.lower())
    tokens = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", normalized)
    cjk = [token for token in tokens if len(token) == 1 and "\u4e00" <= token <= "\u9fff"]
    bigrams = ["".join(cjk[index:index + 2]) for index in range(len(cjk) - 1)]
    return tokens + bigrams


@lru_cache(maxsize=1)
def _get_bm25_index() -> tuple[Any, list[Document]]:
    """从持久化 Chroma 读取同一份 chunk，避免维护第二份文本索引。"""
    try:
        from rank_bm25 import BM25Okapi
    except ImportError as exc:
        raise RuntimeError("缺少 rank-bm25 依赖，请安装 requirements.txt") from exc

    manifest = _persist_directory() / "build_manifest.json"
    if not manifest.is_file():
        raise TCMRAGNotReadyError("中医 PDF 知识库尚未构建，无法建立 BM25 检索器")
    payload = _create_vector_store().get(include=["documents", "metadatas"])
    documents = [
        Document(page_content=content, metadata=metadata or {})
        for content, metadata in zip(payload.get("documents", []), payload.get("metadatas", []), strict=True)
        if content
    ]
    if not documents:
        raise TCMRAGNotReadyError("Chroma 中没有可供 BM25 检索的中医文本块")
    return BM25Okapi([_tokenize_for_bm25(document.page_content) for document in documents]), documents


def _vector_candidates(vector_store: Any, query: str, limit: int) -> list[Document]:
    return vector_store.similarity_search(query, k=limit)


def _bm25_candidates(index: Any, documents: list[Document], query: str, limit: int) -> list[Document]:
    scores = index.get_scores(_tokenize_for_bm25(query))
    ranked_indexes = sorted(range(len(documents)), key=lambda item: float(scores[item]), reverse=True)
    return [documents[index] for index in ranked_indexes[:limit] if scores[index] > 0]


def _rrf_fuse(rankings: list[list[Document]], limit: int) -> list[Document]:
    """用 Reciprocal Rank Fusion 合并多路排序，保留各检索器的互补性。"""
    scores: dict[str, float] = {}
    documents: dict[str, Document] = {}
    for ranking in rankings:
        for position, document in enumerate(ranking, start=1):
            source_id = str(document.metadata.get("source_id") or hashlib.sha1(document.page_content.encode()).hexdigest())
            documents[source_id] = document
            scores[source_id] = scores.get(source_id, 0.0) + 1 / (60 + position)
    return [documents[source_id] for source_id in sorted(scores, key=scores.get, reverse=True)[:limit]]


def _rerank_candidates(query: str, candidates: list[Document], limit: int) -> tuple[list[Document], bool]:
    """使用 DashScope rerank 对融合候选精排；服务失败时安全降级为 RRF 顺序。"""
    if not _rerank_enabled() or not candidates:
        return candidates[:limit], False
    try:
        import dashscope

        response = dashscope.TextReRank.call(
            model=os.getenv("TCM_RAG_RERANK_MODEL", "qwen3-rerank"),
            query=query,
            documents=[candidate.page_content for candidate in candidates],
            top_n=limit,
            return_documents=False,
            api_key=os.getenv("DASHSCOPE_API_KEY"),
        )
        if getattr(response, "status_code", None) != 200:
            raise RuntimeError(f"DashScope rerank status={getattr(response, 'status_code', 'unknown')}")
        return [candidates[item.index] for item in response.output.results], True
    except Exception as exc:
        logger.warning("tcm_rag_stage=rerank_fallback error_type=%s", type(exc).__name__)
        return candidates[:limit], False


def retrieve_tcm_wellness_knowledge_with_trace(
    query: str,
    rewritten_query: str | None = None,
    limit: int = 3,
) -> TCMRetrievalResult:
    """原查询与重写查询并行执行向量/BM25 召回，RRF 融合后再精排。"""
    started_at = time.perf_counter()
    original_query = query.strip()
    rewritten_query = (rewritten_query or original_query).strip()
    vector_limit = _positive_int("TCM_RAG_VECTOR_CANDIDATES", 8)
    bm25_limit = _positive_int("TCM_RAG_BM25_CANDIDATES", 8)
    rerank_limit = _positive_int("TCM_RAG_RERANK_CANDIDATES", 12)
    query_variants = [original_query] if rewritten_query == original_query else [original_query, rewritten_query]
    # 依赖和本地索引先在主线程初始化，规避并发首次 import Chroma/NumPy 的竞争。
    vector_store = _create_vector_store()
    bm25_index, bm25_documents = _get_bm25_index()

    with ThreadPoolExecutor(max_workers=len(query_variants) * 2) as executor:
        jobs = [
            executor.submit(_vector_candidates, vector_store, variant, vector_limit)
            for variant in query_variants
        ] + [
            executor.submit(_bm25_candidates, bm25_index, bm25_documents, variant, bm25_limit)
            for variant in query_variants
        ]
        rankings = [job.result() for job in jobs]

    vector_count = sum(len(ranking) for ranking in rankings[:len(query_variants)])
    bm25_count = sum(len(ranking) for ranking in rankings[len(query_variants):])
    fused = _rrf_fuse(rankings, limit=rerank_limit)
    rerank_query = original_query if rewritten_query == original_query else f"{original_query}\n检索表达：{rewritten_query}"
    reranked, rerank_applied = _rerank_candidates(rerank_query, fused, limit)
    chunks = [
        RetrievedTCMChunk(
            source_id=str(document.metadata.get("source_id", "tcm-pdf:unknown")),
            title=str(document.metadata.get("source_title", SOURCE_TITLE)),
            content=document.page_content,
            page_start=document.metadata.get("source_page_start"),
            page_end=document.metadata.get("source_page_end"),
        )
        for document in reranked
    ]
    trace = TCMRetrievalTrace(
        original_query=original_query,
        rewritten_query=rewritten_query,
        vector_candidates=vector_count,
        bm25_candidates=bm25_count,
        fused_candidates=len(fused),
        rerank_applied=rerank_applied,
        elapsed_ms=round((time.perf_counter() - started_at) * 1000),
    )
    logger.info(
        "tcm_rag_stage=retrieval vector_candidates=%s bm25_candidates=%s fused_candidates=%s rerank=%s elapsed_ms=%s",
        trace.vector_candidates, trace.bm25_candidates, trace.fused_candidates,
        trace.rerank_applied, trace.elapsed_ms,
    )
    return TCMRetrievalResult(chunks=chunks, trace=trace)


def retrieve_tcm_wellness_knowledge(query: str, limit: int = 3) -> list[RetrievedTCMChunk]:
    """兼容直接检索入口；未提供重写查询时仍执行向量 + BM25 + rerank。"""
    return retrieve_tcm_wellness_knowledge_with_trace(query=query, limit=limit).chunks
