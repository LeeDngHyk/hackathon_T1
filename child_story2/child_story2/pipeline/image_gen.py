"""
Replicate 기반 이미지 생성 (google/nano-banana).

크레딧 $5 미만 시 분당 6요청·동시 1개 제한. 429 메시지의 'resets in ~Xs' 를
파싱해 정확히 그 시간만큼 대기.
"""
from __future__ import annotations

import asyncio
import re
from pathlib import Path

import httpx
import replicate

from pipeline.config import (
    ART_STYLE_LOCK,
    CHARACTER_REF_IMAGES,
    IMAGE_MAX_RETRIES,
    REPLICATE_API_TOKEN,
)


REPLICATE_IMAGE_MODEL = "google/nano-banana"


def _ensure_key() -> None:
    if not REPLICATE_API_TOKEN:
        raise RuntimeError(
            "REPLICATE_API_TOKEN이 설정되지 않았습니다.\n"
            "https://replicate.com/account/api-tokens 에서 발급 후 .env에 추가하세요."
        )


def _collect_ref_paths(characters_present: list[str]) -> list[Path]:
    paths: list[Path] = []
    for cid in characters_present:
        p = CHARACTER_REF_IMAGES.get(cid)
        if p and p.exists():
            paths.append(p)
    return paths


def _parse_retry_seconds(err_msg: str, default: float = 15.0) -> float:
    m = re.search(r"resets?\s+in\s+~?(\d+)\s*s", err_msg, re.IGNORECASE)
    if m:
        return float(m.group(1)) + 2.0
    return default


def _extract_url(output) -> str:
    if isinstance(output, list) and output:
        output = output[0]
    url_str = str(output)
    if url_str.startswith("http"):
        return url_str
    if hasattr(output, "url"):
        u = output.url
        if callable(u):
            u = u()
        u = str(u)
        if u.startswith("http"):
            return u
    raise RuntimeError(f"Replicate 응답에서 이미지 URL 추출 실패: {output!r}")


def _submit_and_wait(prompt: str, ref_paths: list[Path]) -> str:
    ref_files = [open(p, "rb") for p in ref_paths]
    try:
        input_data = {"prompt": prompt, "output_format": "png"}
        if ref_files:
            input_data["image_input"] = ref_files
        output = replicate.run(REPLICATE_IMAGE_MODEL, input=input_data)
        return _extract_url(output)
    finally:
        for f in ref_files:
            f.close()


async def generate_image(
    prompt: str,
    output_path: Path,
    characters_present: list[str] | None = None,
) -> Path:
    _ensure_key()
    if output_path.exists():
        print(f"[image] cache hit: {output_path.name}")
        return output_path

    ref_paths = _collect_ref_paths(characters_present or [])
    full_prompt = f"{prompt.strip()}\n\nStyle: {ART_STYLE_LOCK}"
    if ref_paths:
        full_prompt = (
            "Use the reference images to maintain exact character appearance "
            "(same clothing, hair, facial features).\n\n" + full_prompt
        )

    image_url = None
    for attempt in range(IMAGE_MAX_RETRIES):
        try:
            image_url = await asyncio.to_thread(
                _submit_and_wait, full_prompt, ref_paths
            )
            break
        except Exception as e:
            msg = str(e)
            # 재시도 불가 (429는 재시도 가능이라 제외)
            if any(k in msg for k in ("401", "403", "Insufficient credit", "402")):
                raise RuntimeError(f"Replicate 접근 거부: {msg}") from e
            # 429 → 응답 안의 대기 시간 파싱
            if "429" in msg or "throttled" in msg.lower() or "rate limit" in msg.lower():
                wait = _parse_retry_seconds(msg, default=15.0)
                print(f"[image] rate limit, {wait:.0f}s 대기 ({attempt + 1}/{IMAGE_MAX_RETRIES}): {output_path.name}")
            else:
                wait = min(2 ** attempt, 30)
                print(f"[image] 에러: {e!r}, {wait}s 대기 후 재시도 ({attempt + 1}/{IMAGE_MAX_RETRIES})")
            if attempt == IMAGE_MAX_RETRIES - 1:
                raise
            await asyncio.sleep(wait)

    async with httpx.AsyncClient() as client:
        r = await client.get(image_url, timeout=120)
        r.raise_for_status()
        output_path.write_bytes(r.content)

    print(f"[image] 저장: {output_path.name}")
    return output_path