"""
Typecast (ssfm-v30) TTS 생성.

씬 하나는 [내레이션] → [대사1] → [대사2] → ... 를 이어 붙인 하나의 오디오 트랙.
- 각 라인은 speaker_id로 다른 voice_id 사용
- voice_direction 힌트 → Typecast emotion_preset + intensity 매핑
- 라인 사이에 짧은 무음 삽입 (INTER_LINE_PAUSE_MS)
- ffmpeg concat demuxer로 하나의 mp3로 이어붙임
"""
from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

import httpx

from pipeline.config import (
    EMOTION_BY_TONE,
    FFMPEG_BIN,
    INTER_LINE_PAUSE_MS,
    TMP_DIR,
    TTS_AUDIO_FORMAT,
    TTS_LANGUAGE,
    TTS_MAX_CONCURRENT,
    TTS_MAX_RETRIES,
    TTS_MODEL,
    TYPECAST_API_KEY,
    TYPECAST_TTS_URL,
    VOICE_IDS,
)


_tts_semaphore: asyncio.Semaphore | None = None


def _get_tts_semaphore() -> asyncio.Semaphore:
    global _tts_semaphore
    if _tts_semaphore is None:
        _tts_semaphore = asyncio.Semaphore(TTS_MAX_CONCURRENT)
    return _tts_semaphore


TONE_KEYWORDS = {
    "excited": ["신나", "밝고", "설레", "기대", "즐거", "행복", "환한", "기뻐", "명랑"],
    "sad":     ["울", "슬프", "속상", "눈물", "울먹", "안타까", "서러"],
    "angry":   ["화난", "화가", "격하게", "짜증", "분노"],
    "gentle":  ["부드럽", "다정", "따뜻", "차분", "안정", "격려", "이해", "위로", "다독"],
    "whisper": ["속삭", "조용히", "낮은 목소리"],
    "bright":  ["활기", "당당", "또박또박"],
}


def infer_tone(voice_direction: str) -> str:
    if not voice_direction:
        return "default"
    for tone, keywords in TONE_KEYWORDS.items():
        if any(kw in voice_direction for kw in keywords):
            return tone
    return "default"


def get_voice_id(speaker_id: str) -> str:
    return VOICE_IDS.get(speaker_id, VOICE_IDS["narrator"])


async def synth_line(
    text: str,
    speaker_id: str,
    voice_direction: str,
    output_path: Path,
    client: httpx.AsyncClient,
) -> Path:
    if output_path.exists():
        return output_path

    voice_id = get_voice_id(speaker_id)
    tone = infer_tone(voice_direction)
    emotion = EMOTION_BY_TONE[tone]

    payload = {
        "text": text,
        "voice_id": voice_id,
        "model": TTS_MODEL,
        "prompt": {
            "emotion_type": "preset",
            "emotion_preset": emotion["emotion_preset"],
            "emotion_intensity": emotion["emotion_intensity"],
        },
        "output": {
            "audio_format": TTS_AUDIO_FORMAT,
            "volume": 100,
            "audio_pitch": 0,
            "audio_tempo": 1.0,
        },
    }
    if TTS_LANGUAGE:
        payload["language"] = TTS_LANGUAGE

    headers = {
        "X-API-KEY": TYPECAST_API_KEY,
        "Content-Type": "application/json",
    }

    semaphore = _get_tts_semaphore()

    for attempt in range(TTS_MAX_RETRIES):
        async with semaphore:
            r = await client.post(TYPECAST_TTS_URL, headers=headers, json=payload, timeout=90)

        if r.status_code == 200:
            output_path.write_bytes(r.content)
            return output_path

        if r.status_code == 429 or 500 <= r.status_code < 600:
            retry_after = r.headers.get("Retry-After")
            try:
                wait = float(retry_after) if retry_after else min(2 ** attempt, 30)
            except ValueError:
                wait = min(2 ** attempt, 30)
            print(
                f"[tts] {r.status_code} 응답, {wait:.1f}s 대기 후 재시도 "
                f"({attempt + 1}/{TTS_MAX_RETRIES}): {output_path.name}"
            )
            await asyncio.sleep(wait)
            continue

        if r.status_code == 401:
            raise RuntimeError("Typecast 인증 실패: TYPECAST_API_KEY 확인")
        if r.status_code == 402:
            raise RuntimeError("Typecast 크레딧 부족")
        if r.status_code == 422:
            raise RuntimeError(f"Typecast 요청 파라미터 오류: {r.text}")
        r.raise_for_status()

    raise RuntimeError(
        f"Typecast 재시도 {TTS_MAX_RETRIES}회 후에도 실패: {output_path.name}"
    )


def _silence_file(pause_ms: int) -> Path:
    path = TMP_DIR / f"silence_{pause_ms}ms.mp3"
    if path.exists():
        return path
    seconds = pause_ms / 1000.0
    subprocess.run(
        [
            FFMPEG_BIN, "-y", "-f", "lavfi", "-i",
            "anullsrc=channel_layout=stereo:sample_rate=44100",
            "-t", f"{seconds}", "-q:a", "9", "-acodec", "libmp3lame",
            str(path),
        ],
        check=True, capture_output=True,
    )
    return path


def concat_mp3(inputs: list[Path], output: Path) -> Path:
    list_file = output.with_suffix(".txt")
    list_file.write_text(
        "\n".join(f"file '{p.resolve().as_posix()}'" for p in inputs),
        encoding="utf-8",
    )
    subprocess.run(
        [
            FFMPEG_BIN, "-y", "-f", "concat", "-safe", "0",
            "-i", str(list_file), "-c", "copy", str(output),
        ],
        check=True, capture_output=True,
    )
    list_file.unlink()
    return output


async def synth_scene_audio(scene: dict, output_path: Path) -> Path:
    if output_path.exists():
        print(f"[tts] cache hit: {output_path.name}")
        return output_path

    scene_id = scene["scene_id"]
    line_paths: list[Path] = []

    async with httpx.AsyncClient() as client:
        narration = scene.get("narration", "").strip()
        if narration:
            p = TMP_DIR / f"{scene_id}_narration.{TTS_AUDIO_FORMAT}"
            await synth_line(narration, "narrator", "차분하게", p, client)
            line_paths.append(p)

        for idx, line in enumerate(scene.get("dialogue", []) or []):
            text = line.get("text", "").strip()
            if not text:
                continue
            speaker = line.get("speaker_id", "narrator")
            vd = line.get("voice_direction", "")
            p = TMP_DIR / f"{scene_id}_line_{idx:02d}.{TTS_AUDIO_FORMAT}"
            await synth_line(text, speaker, vd, p, client)
            line_paths.append(p)

    if not line_paths:
        raise RuntimeError(f"{scene_id}: 합성할 텍스트가 없음")

    silence = _silence_file(INTER_LINE_PAUSE_MS)
    interleaved: list[Path] = []
    for i, p in enumerate(line_paths):
        interleaved.append(p)
        if i < len(line_paths) - 1:
            interleaved.append(silence)

    concat_mp3(interleaved, output_path)
    print(f"[tts] 저장: {output_path.name}")
    return output_path


async def synth_quiz_prompt(quiz: dict, output_path: Path) -> Path:
    if output_path.exists():
        return output_path
    text = quiz.get("voice_prompt") or quiz.get("question", "")
    async with httpx.AsyncClient() as client:
        await synth_line(text, "obj_1", "부드럽고 다정하게", output_path, client)
    return output_path
