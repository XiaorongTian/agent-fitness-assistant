"""命令行入口：构建《了不起的中医养生妙招》PDF RAG 索引。"""

from __future__ import annotations

import argparse
import json

from rag.tcm_wellness import build_tcm_wellness_index


def main() -> None:
    parser = argparse.ArgumentParser(description="构建中医 PDF Chroma 知识库")
    parser.add_argument("--reset", action="store_true", help="删除目标 Chroma collection 后重新构建")
    args = parser.parse_args()
    print(json.dumps(build_tcm_wellness_index(reset=args.reset), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
