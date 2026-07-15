"""
스토리 JSON에 씬별 이미지·모션 프롬프트(영어)를 부착.

각 씬의 story_event / characters_present / location / child_emotion을 근거로
Text-to-Image 용 image_prompt_en과 Image-to-Video 용 motion_prompt_en을 채운다.

캐릭터 시각 프리셋(config.CHARACTER_VISUAL_LOCK)은 매 프롬프트에 문자 그대로
삽입되도록 규칙에 명시한다. 이것이 씬 간 캐릭터 일관성의 핵심 장치.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from anthropic import Anthropic

from pipeline.config import (
    ANTHROPIC_API_KEY,
    ART_STYLE_LOCK,
    CHARACTER_VISUAL_LOCK,
    ENRICH_MAX_TOKENS,
    ENRICH_MODEL,
)


ENRICH_SYSTEM_PROMPT = f"""당신은 유아 동화 영상 파이프라인의 시각 프롬프트 설계자입니다.

입력받은 스토리 JSON의 각 씬(메인 + 분기)에 대해 두 개의 영어 프롬프트를 만드세요.

1) image_prompt_en — Text-to-Image용 정지 이미지 프롬프트
2) motion_prompt_en — Image-to-Video용 모션 프롬프트

[규칙]
- 두 프롬프트 모두 반드시 영어로 작성.
- image_prompt_en 조립 순서:
  [등장 캐릭터의 CHARACTER_VISUAL_LOCK 문자열들] + [씬의 행동/표정] +
  [배경 · 시간대] + [구도 · shot type] + [ART_STYLE_LOCK]
  → CHARACTER_VISUAL_LOCK 문자열은 아래 표에서 그대로 복사(패러프레이즈 금지).
- motion_prompt_en 규칙:
  · 시작 프레임(생성된 이미지)의 캐릭터와 배경을 유지하도록 지시.
  · 캐릭터의 작은 동작 1~2개 + 느린 카메라 움직임(slow push-in, gentle pan 등).
  · 새로운 인물·의상·소품 추가 금지, 급격한 컷·변신 금지.
- 사진 참조 언급 금지("as in the photo", "reference image" 등 절대 사용 X).
- 이미지 안에 글자·자막 생성 방지 문구를 image_prompt_en에 포함.

[CHARACTER_VISUAL_LOCK 표 — 문자 그대로 삽입]
{json.dumps(CHARACTER_VISUAL_LOCK, ensure_ascii=False, indent=2)}

[ART_STYLE_LOCK — image_prompt_en 끝에 그대로 붙임]
{ART_STYLE_LOCK}

[출력 형식]
반드시 아래 형식의 유효한 JSON 하나만 반환. 마크다운·설명 없이.
{{
  "scenes": {{
    "S01": {{ "image_prompt_en": "...", "motion_prompt_en": "..." }},
    ...
  }},
  "branch_scenes": {{
    "Q01_A": {{ "image_prompt_en": "...", "motion_prompt_en": "..." }},
    ...
  }}
}}
"""


def build_user_prompt(story: dict[str, Any]) -> str:
    def compact_scene(s: dict[str, Any]) -> dict[str, Any]:
        return {
            "scene_id": s["scene_id"],
            "location": s.get("location", ""),
            "time_of_day": s.get("time_of_day", ""),
            "characters_present": s.get("characters_present", []),
            "story_event": s.get("story_event", ""),
            "child_emotion": s.get("child_emotion", ""),
        }

    payload = {
        "mainline_scenes": [compact_scene(s) for s in story.get("mainline_scenes", [])],
        "branch_scenes": [compact_scene(s) for s in story.get("branch_scenes", [])],
    }
    return (
        "다음 스토리의 각 씬에 대해 image_prompt_en과 motion_prompt_en을 생성하세요.\n\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def strip_code_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        first_nl = t.find("\n")
        if first_nl != -1:
            t = t[first_nl + 1 :]
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip()


def enrich(story: dict[str, Any]) -> dict[str, Any]:
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY가 필요합니다.")

    client = Anthropic()
    resp = client.messages.create(
        model=ENRICH_MODEL,
        max_tokens=ENRICH_MAX_TOKENS,
        system=ENRICH_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_user_prompt(story)}],
    )
    raw = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    prompts = json.loads(strip_code_fence(raw))

    enriched = json.loads(json.dumps(story))
    for s in enriched.get("mainline_scenes", []):
        p = prompts.get("scenes", {}).get(s["scene_id"], {})
        s["image_prompt_en"] = p.get("image_prompt_en", "")
        s["motion_prompt_en"] = p.get("motion_prompt_en", "")
    for s in enriched.get("branch_scenes", []):
        p = prompts.get("branch_scenes", {}).get(s["scene_id"], {})
        s["image_prompt_en"] = p.get("image_prompt_en", "")
        s["motion_prompt_en"] = p.get("motion_prompt_en", "")

    validate_lock_inclusion(enriched)
    return enriched


def validate_lock_inclusion(enriched: dict[str, Any]) -> None:
    def check(scenes: list[dict[str, Any]]) -> list[str]:
        errs = []
        for s in scenes:
            prompt = s.get("image_prompt_en", "")
            for cid in s.get("characters_present", []):
                lock = CHARACTER_VISUAL_LOCK.get(cid, "")
                if not lock:
                    continue
                head = " ".join(lock.split()[:15])
                if head not in prompt:
                    errs.append(f"{s['scene_id']}: '{cid}' lock 문자열 누락")
        return errs

    errors = check(enriched.get("mainline_scenes", [])) + check(enriched.get("branch_scenes", []))
    if errors:
        print("[enrich] 경고: 캐릭터 락 문자열 검증 실패")
        for e in errors:
            print(f"  - {e}")


def enrich_file(input_path: Path, output_path: Path) -> None:
    story = json.loads(input_path.read_text(encoding="utf-8"))
    enriched = enrich(story)
    output_path.write_text(
        json.dumps(enriched, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[enrich] 저장: {output_path}")


if __name__ == "__main__":
    import sys
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("output_story.json")
    dst = src.with_name("enriched_story.json")
    enrich_file(src, dst)
