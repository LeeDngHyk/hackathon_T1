"""
퀴즈 음성 입력 → 4지선다 선택지 매칭.

STT: Replicate Whisper (그대로)
임베딩: 로컬 sentence-transformers (OpenAI 크레딧 불필요, 완전 무료)

첫 실행 시 모델(~120MB) 자동 다운로드. 이후 오프라인 동작.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

import replicate
import numpy as np
from sentence_transformers import SentenceTransformer

from pipeline.config import (
    EMBEDDING_MATCH_THRESHOLD,
    REPLICATE_API_TOKEN,
    REPLICATE_STT_MODEL,
)


# ═════════════════════════════════════════════════════════════
# 로컬 임베딩 모델 (한국어 포함 50+ 언어 지원)
# ═════════════════════════════════════════════════════════════
# paraphrase-multilingual-MiniLM-L12-v2:
#   - 384차원, ~120MB, 50+개 언어
#   - 한국어 유아 발화 매칭에 충분
#   - 첫 로드 시 ~/.cache/huggingface/ 에 자동 다운로드
_LOCAL_EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"[embedding] 로컬 모델 로드 중… ({_LOCAL_EMBEDDING_MODEL})")
        _model = SentenceTransformer(_LOCAL_EMBEDDING_MODEL)
        print("[embedding] 로드 완료")
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """여러 문장을 한 번에 임베딩 → (N, 384)"""
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    embeddings = _get_model().encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return embeddings.astype(np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


# ═════════════════════════════════════════════════════════════
# STT (Replicate Whisper)
# ═════════════════════════════════════════════════════════════
async def transcribe_audio(audio_path: Path, language: str = "ko") -> str:
    if not REPLICATE_API_TOKEN:
        raise RuntimeError("REPLICATE_API_TOKEN 필요")

    def _run():
        with open(audio_path, "rb") as f:
            output = replicate.run(
                REPLICATE_STT_MODEL,
                input={"audio": f, "language": language, "model": "large-v3"},
            )
        if isinstance(output, dict):
            return output.get("transcription", output.get("text", ""))
        return str(output).strip()

    return await asyncio.to_thread(_run)


# ═════════════════════════════════════════════════════════════
# 선택지 사전 벡터화
# ═════════════════════════════════════════════════════════════
@dataclass
class ChoiceIndex:
    option_id: str
    branch_scene_id: str
    is_preferred: bool
    anchor_vec: np.ndarray
    accepted_vecs: np.ndarray
    excluded_vecs: np.ndarray


def build_quiz_index(quiz: dict) -> list[ChoiceIndex]:
    indices: list[ChoiceIndex] = []
    for c in quiz["choices"]:
        anchor = embed_texts([c["semantic_anchor"]])[0]
        accepted = embed_texts(c.get("accepted_utterances") or [c["semantic_anchor"]])
        excluded_texts = c.get("excluded_meanings") or []
        excluded = embed_texts(excluded_texts) if excluded_texts else np.zeros((0, anchor.shape[0]), dtype=np.float32)
        indices.append(
            ChoiceIndex(
                option_id=c["option_id"],
                branch_scene_id=c["branch_scene_id"],
                is_preferred=bool(c.get("is_preferred_choice")),
                anchor_vec=anchor,
                accepted_vecs=accepted,
                excluded_vecs=excluded,
            )
        )
    return indices


# ═════════════════════════════════════════════════════════════
# 매칭
# ═════════════════════════════════════════════════════════════
@dataclass
class MatchResult:
    matched_option_id: str
    matched_branch_scene_id: str
    confidence: float
    scores: dict[str, float]
    transcript: str
    is_low_confidence: bool


def match_utterance_to_choice(
    utterance_text: str,
    choice_index: list[ChoiceIndex],
) -> MatchResult:
    u = embed_texts([utterance_text])[0]
    scores: dict[str, float] = {}
    for ci in choice_index:
        pos_scores = [cosine(u, ci.anchor_vec)] + [cosine(u, v) for v in ci.accepted_vecs]
        neg_scores = [cosine(u, v) for v in ci.excluded_vecs] if len(ci.excluded_vecs) else [0.0]
        scores[ci.option_id] = max(pos_scores) - max(0.0, max(neg_scores) - 0.5)

    best_id = max(scores, key=scores.get)  # type: ignore
    best = next(c for c in choice_index if c.option_id == best_id)
    conf = scores[best_id]

    return MatchResult(
        matched_option_id=best_id,
        matched_branch_scene_id=best.branch_scene_id,
        confidence=conf,
        scores=scores,
        transcript=utterance_text,
        is_low_confidence=conf < EMBEDDING_MATCH_THRESHOLD,
    )


async def match_audio_to_choice(
    audio_path: Path,
    choice_index: list[ChoiceIndex],
) -> MatchResult:
    transcript = await transcribe_audio(audio_path)
    if not transcript:
        return MatchResult("", "", 0.0, {}, "", True)
    return match_utterance_to_choice(transcript, choice_index)