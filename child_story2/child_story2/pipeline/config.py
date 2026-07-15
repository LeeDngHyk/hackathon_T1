"""
파이프라인 설정.

이미지: Google Gemini 2.5 Flash Image (google-genai SDK)
영상:  Replicate Kling v1.6 (replicate SDK)
STT:   Replicate Whisper
TTS:   Typecast ssfm-v30
스토리: Claude Sonnet 4.5
임베딩: OpenAI text-embedding-3-small
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────
# 경로
# ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"
SCENES_DIR = OUTPUT_DIR / "scenes"
BRANCHES_DIR = OUTPUT_DIR / "branches"
TMP_DIR = OUTPUT_DIR / "tmp"
UPLOADS_DIR = BASE_DIR / "uploads"

for d in (OUTPUT_DIR, SCENES_DIR, BRANCHES_DIR, TMP_DIR, UPLOADS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# API 키
# ─────────────────────────────────────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")             # Gemini (이미지)
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")   # Kling 영상, Whisper STT
TYPECAST_API_KEY = os.getenv("TYPECAST_API_KEY")         # TTS
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")       # 스토리·프롬프트 생성
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")             # 퀴즈 임베딩

# ─────────────────────────────────────────────────────────────
# 모델
# ─────────────────────────────────────────────────────────────
STORY_MODEL = "claude-sonnet-4-5"
STORY_MAX_TOKENS = 16000

ENRICH_MODEL = "claude-sonnet-4-5"
ENRICH_MAX_TOKENS = 16000

# Google Gemini 이미지 생성
GEMINI_IMAGE_MODEL = "gemini-2.5-flash-image"
IMAGE_SIZE = {"width": 1024, "height": 768}

# Replicate 영상 (Image-to-Video)
REPLICATE_VIDEO_MODEL = "kwaivgi/kling-v1.6-standard"
# 대안: "minimax/video-01"  (더 저렴)
#       "wavespeedai/wan-2.1-i2v-480p"  (저렴, 빠름)
VIDEO_DURATION_SEC = 5
VIDEO_POLL_INTERVAL_SEC = 10
VIDEO_POLL_TIMEOUT_SEC = 600

# Replicate STT (퀴즈 음성)
REPLICATE_STT_MODEL = "openai/whisper:8099696689d249cf8b122d833c36a428d2d9ea8281b46b4b23ca54e1f5013f12"

# TTS (Typecast)
TYPECAST_TTS_URL = "https://api.typecast.ai/v1/text-to-speech"
TTS_MODEL = "ssfm-v30"
TTS_AUDIO_FORMAT = "mp3"
TTS_LANGUAGE = "kor"

# 임베딩 (퀴즈)
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_MATCH_THRESHOLD = 0.35

# ─────────────────────────────────────────────────────────────
# 스토리 기본값
# ─────────────────────────────────────────────────────────────
STORY_DEFAULTS = {
    "scene_count": 6,
    "quiz_count": 1,
    "age_range": "3-6세",
    "tone": (
        "따뜻하고 다정한 유아 동화 톤. 짧은 문장, 쉬운 어휘. "
        "훈계·명령·부정문 최소화. 조력자는 지시하지 않고 질문·감탄만."
    ),
    "art_style_note": "부드러운 수채화 그림책 스타일",
}

# ─────────────────────────────────────────────────────────────
# 그림체 잠금
# ─────────────────────────────────────────────────────────────
ART_STYLE_LOCK = (
    "soft watercolor children's book illustration, pastel color palette, "
    "gentle warm lighting, rounded shapes, cozy storybook atmosphere, "
    "hand-painted texture, wholesome and calm mood, "
    "no text, no letters, no watermark"
)

NEGATIVE_PROMPT = (
    "text, subtitle, letters, watermark, logo, "
    "extra fingers, deformed hands, scary, dark, gloomy, "
    "photo-realistic, 3d render"
)

# ─────────────────────────────────────────────────────────────
# 캐릭터 시각 프리셋
# ─────────────────────────────────────────────────────────────
CHARACTER_VISUAL_LOCK = {
    "protagonist": (
        "a cheerful 5-year-old Korean girl named Jiu, "
        "chin-length black bob hair with straight bangs, round face, "
        "big expressive eyes, wearing a yellow t-shirt and denim overalls, "
        "white socks"
    ),
    "obj_1": (
        "Toto, a small plush rabbit doll, cream-colored soft fur, "
        "long floppy ears, pink stitched nose, gentle friendly face, "
        "sitting upright, about the size of a child's forearm"
    ),
    "sibling": (
        "Darami, a small squirrel plush doll, warm brown fur, "
        "fluffy striped tail, big round black eyes, tiny paws, "
        "smaller than Toto"
    ),
    "narrator": "",
}

# 참조 이미지 (부모 업로드 — 없어도 동작함)
CHARACTER_REF_IMAGES = {
    "protagonist": UPLOADS_DIR / "child.jpg",
    "obj_1": UPLOADS_DIR / "obj_1.jpg",
    "sibling": UPLOADS_DIR / "sibling.jpg",
}

# ─────────────────────────────────────────────────────────────
# Typecast voice_id
# ─────────────────────────────────────────────────────────────
VOICE_IDS = {
    "narrator": "tc_6699eb3849dfac016c29444c",
    "protagonist": "tc_6699eb3849dfac016c29444c",
    "obj_1": "tc_6699eb3849dfac016c29444c",
    "sibling": "tc_6699eb3849dfac016c29444c",
}

EMOTION_BY_TONE = {
    "default":  {"emotion_preset": "normal",   "emotion_intensity": 1.0},
    "excited":  {"emotion_preset": "happy",    "emotion_intensity": 1.3},
    "sad":      {"emotion_preset": "sad",      "emotion_intensity": 1.2},
    "gentle":   {"emotion_preset": "tonedown", "emotion_intensity": 1.0},
    "angry":    {"emotion_preset": "angry",    "emotion_intensity": 1.1},
    "whisper":  {"emotion_preset": "whisper",  "emotion_intensity": 1.0},
    "bright":   {"emotion_preset": "toneup",   "emotion_intensity": 1.1},
}

# ─────────────────────────────────────────────────────────────
# 파이프라인 파라미터
# ─────────────────────────────────────────────────────────────
MAX_PARALLEL_JOBS = 4
TTS_MAX_CONCURRENT = 2
TTS_MAX_RETRIES = 5
IMAGE_MAX_RETRIES = 3
VIDEO_MAX_RETRIES = 3
INTER_LINE_PAUSE_MS = 400
FFMPEG_BIN = os.getenv("FFMPEG_BIN", "ffmpeg")
FFPROBE_BIN = os.getenv("FFPROBE_BIN", "ffprobe")