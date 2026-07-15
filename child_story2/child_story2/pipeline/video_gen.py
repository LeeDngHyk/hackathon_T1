"""
Replicate Kling v1.6 기반 image-to-video.

이미지 모듈과 동일한 429 파싱 재시도 로직.
"""
from __future__ import annotations

import asyncio
import re
import time
from pathlib import Path

import httpx
import replicate

from pipeline.config import (
    REPLICATE_API_TOKEN,
    REPLICATE_VIDEO_MODEL,
    VIDEO_DURATION_SEC,
    VIDEO_MAX_RETRIES,
    VIDEO_POLL_INTERVAL_SEC,
    VIDEO_POLL_TIMEOUT_SEC,
)


def _ensure_key() -> None:
    if not REPLICATE_API_TOKEN:
        raise RuntimeError(
            "REPLICATE_API_TOKEN이 설정되지 않았습니다.\n"
            "https://replicate.com/account/api-tokens 에서 발급 후 .env에 추가하세요."
        )


def _parse_retry_seconds(err_msg: str, default: float = 15.0) -> float:
    m = re.search(r"resets?\s+in\s+~?(\d+)\s*s", err_msg, re.IGNORECASE)
    if m:
        return float(m.group(1)) + 2.0
    return default


def _submit_and_wait(image_path: Path, motion_prompt: str):
    with open(image_path, "rb") as f:
        prediction = replicate.predictions.create(
            model=REPLICATE_VIDEO_MODEL,
            input={
                "prompt": motion_prompt,
                "start_image": f,
                "duration": VIDEO_DURATION_SEC,
                "cfg_scale": 0.5,
                "aspect_ratio": "1:1",
                "negative_prompt": (
                    "sudden movement, character change, extra characters, "
                    "distorted face, morphing, fast camera, text appearing, logo"
                ),
            },
        )
    print(f"[video] {image_path.stem} 제출됨 (prediction {prediction.id})")

    elapsed = 0
    while prediction.status not in ("succeeded", "failed", "canceled"):
        if elapsed >= VIDEO_POLL_TIMEOUT_SEC:
            raise RuntimeError(f"Replicate 타임아웃 ({VIDEO_POLL_TIMEOUT_SEC}s): {prediction.id}")
        time.sleep(VIDEO_POLL_INTERVAL_SEC)
        elapsed += VIDEO_POLL_INTERVAL_SEC
        prediction.reload()

    if prediction.status != "succeeded":
        raise RuntimeError(f"Replicate prediction 실패: {prediction.status} — {prediction.error}")
    return prediction


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
    raise RuntimeError(f"Replicate 응답에서 video URL 추출 실패: {output!r}")


async def generate_video(
    image_path: Path,
    motion_prompt: str,
    output_path: Path,
) -> Path:
    _ensure_key()
    if output_path.exists():
        print(f"[video] cache hit: {output_path.name}")
        return output_path

    prediction = None
    for attempt in range(VIDEO_MAX_RETRIES):
        try:
            prediction = await asyncio.to_thread(
                _submit_and_wait, image_path, motion_prompt
            )
            break
        except Exception as e:
            msg = str(e)
            if any(k in msg for k in ("401", "403", "Insufficient credit", "402")):
                raise RuntimeError(f"Replicate 접근 거부: {msg}") from e
            if "429" in msg or "throttled" in msg.lower() or "rate limit" in msg.lower():
                wait = _parse_retry_seconds(msg, default=15.0)
                print(f"[video] rate limit, {wait:.0f}s 대기 ({attempt + 1}/{VIDEO_MAX_RETRIES}): {output_path.name}")
            else:
                wait = min(2 ** attempt, 30)
                print(f"[video] 에러: {e!r}, {wait}s 대기 후 재시도 ({attempt + 1}/{VIDEO_MAX_RETRIES})")
            if attempt == VIDEO_MAX_RETRIES - 1:
                raise
            await asyncio.sleep(wait)

    video_url = _extract_url(prediction.output)

    async with httpx.AsyncClient() as client:
        r = await client.get(video_url, timeout=300)
        r.raise_for_status()
        output_path.write_bytes(r.content)

    print(f"[video] 저장: {output_path.name}")
    return output_path