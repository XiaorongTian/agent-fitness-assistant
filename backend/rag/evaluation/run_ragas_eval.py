"""批量评测中医 RAG：Faithfulness、Context Precision、Context Recall。"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from agents.tcm_agent import generate_tcm_wellness_answer
from rag.tcm_wellness import retrieve_tcm_wellness_knowledge_with_trace
from tcm.knowledge import KnowledgeChunk
from tcm.query_rewriter import rewrite_tcm_retrieval_query


EVALUATION_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_PATH = EVALUATION_ROOT / "tcm_retrieval_eval.jsonl"
DEFAULT_OUTPUT_DIRECTORY = EVALUATION_ROOT / "results"


def _load_ragas() -> tuple[Any, Any, Any, Any]:
    """延迟导入，避免 Ragas 依赖影响后端 API 的正常启动。"""
    try:
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics.collections import ContextPrecision, ContextRecall, Faithfulness
    except ModuleNotFoundError as exc:
        if exc.name == "_lzma":
            raise RuntimeError(
                "当前 Python 缺少 _lzma 扩展，Ragas 的 datasets 依赖无法加载。"
                "请使用带 xz/lzma 支持的 Python 重新创建虚拟环境后再运行评测。"
            ) from exc
        raise RuntimeError("缺少 Ragas，请执行 pip install -r requirements.txt") from exc
    return LangchainLLMWrapper, Faithfulness, ContextPrecision, ContextRecall


def _load_records(path: Path, limit: int | None) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise ValueError(f"评测集为空：{path}")
    return records[:limit] if limit else records


def _output_to_text(output: dict[str, Any]) -> str:
    """将生产 Agent 的结构化输出还原成待评判的自然语言答案。"""
    actions = output.get("actions", [])
    action_lines = [f"{item['title']}：{item['detail']}" for item in actions]
    return "\n".join(
        part for part in [
            output.get("title", ""), output.get("summary", ""), *action_lines,
            output.get("safety_notice", ""),
        ] if part
    )


def _id_precision(retrieved_ids: list[str], reference_ids: list[str]) -> float:
    return len(set(retrieved_ids) & set(reference_ids)) / len(retrieved_ids) if retrieved_ids else 0.0


def _id_recall(retrieved_ids: list[str], reference_ids: list[str]) -> float:
    return len(set(retrieved_ids) & set(reference_ids)) / len(reference_ids) if reference_ids else 0.0


async def _evaluate_record(
    record: dict[str, Any],
    semaphore: asyncio.Semaphore,
    faithfulness: Any,
    context_precision: Any,
    context_recall: Any,
) -> dict[str, Any]:
    async with semaphore:
        user_context: dict[str, Any] = {}
        question = record["user_input"]
        rewritten_query = await rewrite_tcm_retrieval_query(question, user_context)
        retrieval = retrieve_tcm_wellness_knowledge_with_trace(
            query=question, rewritten_query=rewritten_query, limit=3
        )
        chunks = [
            KnowledgeChunk(source_id=item.source_id, title=item.title, content=item.content)
            for item in retrieval.chunks
        ]
        answer = await generate_tcm_wellness_answer(question, user_context, chunks)
        response = _output_to_text(answer)
        retrieved_contexts = [chunk.content for chunk in chunks]
        retrieved_ids = [chunk.source_id for chunk in chunks]

        faithfulness_score, precision_score, recall_score = await asyncio.gather(
            faithfulness.ascore(
                user_input=question,
                response=response,
                retrieved_contexts=retrieved_contexts,
            ),
            context_precision.ascore(
                user_input=question,
                reference=record["reference"],
                retrieved_contexts=retrieved_contexts,
            ),
            context_recall.ascore(
                user_input=question,
                reference=record["reference"],
                retrieved_contexts=retrieved_contexts,
            ),
        )
        return {
            "id": record["id"],
            "category": record["category"],
            "user_input": question,
            "rewritten_query": rewritten_query,
            "response": response,
            "retrieved_context_ids": retrieved_ids,
            "reference_context_ids": record["reference_context_ids"],
            "faithfulness": float(faithfulness_score.value),
            "context_precision": float(precision_score.value),
            "context_recall": float(recall_score.value),
            "id_context_precision": _id_precision(retrieved_ids, record["reference_context_ids"]),
            "id_context_recall": _id_recall(retrieved_ids, record["reference_context_ids"]),
            "retrieval_elapsed_ms": retrieval.trace.elapsed_ms,
            "rerank_applied": retrieval.trace.rerank_applied,
        }


def _summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = (
        "faithfulness", "context_precision", "context_recall",
        "id_context_precision", "id_context_recall", "retrieval_elapsed_ms",
    )
    summary = {"sample_count": len(results)}
    summary.update({
        f"avg_{metric}": round(statistics.fmean(item[metric] for item in results), 4)
        for metric in metric_names
    })
    return summary


async def run(dataset_path: Path, output_path: Path, limit: int | None, concurrency: int) -> dict[str, Any]:
    LangchainLLMWrapper, Faithfulness, ContextPrecision, ContextRecall = _load_ragas()
    from agents.chat_agent import get_chat_model

    judge_llm = LangchainLLMWrapper(get_chat_model())
    faithfulness = Faithfulness(llm=judge_llm)
    context_precision = ContextPrecision(llm=judge_llm)
    context_recall = ContextRecall(llm=judge_llm)
    semaphore = asyncio.Semaphore(concurrency)
    records = _load_records(dataset_path, limit)
    results = await asyncio.gather(*(
        _evaluate_record(record, semaphore, faithfulness, context_precision, context_recall)
        for record in records
    ))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(json.dumps(result, ensure_ascii=False) for result in results) + "\n",
        encoding="utf-8",
    )
    summary = _summary(results)
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"summary": summary, "output_path": str(output_path), "summary_path": str(summary_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="批量执行中医 RAG Ragas 评测")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIRECTORY / "tcm_ragas_results.jsonl")
    parser.add_argument("--limit", type=int, default=10, help="先小批量试跑，例如 10")
    parser.add_argument("--concurrency", type=int, default=int(os.getenv("TCM_RAG_EVAL_CONCURRENCY", "2")))
    args = parser.parse_args()
    if args.concurrency < 1:
        raise SystemExit("--concurrency 必须大于等于 1")
    try:
        result = asyncio.run(run(args.dataset, args.output, args.limit, args.concurrency))
    except Exception as exc:
        print(f"Ragas 评测失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()



# python -m rag.evaluation.run_ragas_eval --limit 10 --concurrency 1