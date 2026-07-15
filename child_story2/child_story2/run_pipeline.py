"""
파이프라인 실행 진입점.

사용법:
    python run_pipeline.py                  # 아래 데모 입력으로 실행
    python run_pipeline.py inputs.json      # JSON 파일에서 입력 읽기

inputs.json 예시:
{
  "child_profile": {"name":"지우","age":5,"gender":"female",
                    "traits":["호기심 많음","블록놀이 좋아함"]},
  "parent_diary": "지우가 블록 놀고 나서 정리를 안 해요...",
  "parent_goal": "블록 놀이 후 스스로 정리하는 습관을 만들고 싶어요"
}
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from pipeline.orchestrator import run_from_parent_input


def load_inputs(argv: list[str]) -> dict:
    if len(argv) > 1:
        return json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    # 데모 입력
    return {
        "child_profile": {
            "name": "지우",
            "age": 5,
            "gender": "female",
            "traits": ["호기심 많음", "부끄러움 잘 탐", "블록놀이 좋아함"],
        },
        "parent_diary": (
            "지우가 요즘 블록으로 성을 만들고 나서 정리를 안 해요. "
            "저녁마다 거실 바닥이 블록 천지라 제가 대신 치우게 돼요. "
            "혼내면 울어버려서 어떻게 알려줄지 고민이에요."
        ),
        "parent_goal": "블록 놀이 후에 스스로 정리하는 습관을 만들고 싶어요.",
    }


def main() -> int:
    inputs = load_inputs(sys.argv)
    asyncio.run(
        run_from_parent_input(
            child_profile=inputs["child_profile"],
            parent_diary=inputs["parent_diary"],
            parent_goal=inputs["parent_goal"],
            work_dir=Path(__file__).resolve().parent,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
