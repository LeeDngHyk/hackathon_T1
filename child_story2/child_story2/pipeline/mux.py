"""
ffmpeg 기반 영상+오디오 합성.

- I2V 무음 영상: 5초 고정 (VIDEO_DURATION_SEC)
- Typecast TTS 오디오: 씬당 5~25초 (가변)
→ 오디오 길이만큼 영상을 loop, -shortest로 오디오 길이에 맞춰 종료.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from pipeline.config import FFMPEG_BIN, FFPROBE_BIN


def probe_duration(path: Path) -> float:
    r = subprocess.run(
        [
            FFPROBE_BIN, "-v", "error", "-show_entries", "format=duration",
            "-of", "json", str(path),
        ],
        capture_output=True, text=True, check=True,
    )
    return float(json.loads(r.stdout)["format"]["duration"])


def mux_scene(video_path: Path, audio_path: Path, output_path: Path) -> Path:
    if output_path.exists():
        print(f"[mux] cache hit: {output_path.name}")
        return output_path

    audio_dur = probe_duration(audio_path)
    video_dur = probe_duration(video_path)
    loops_needed = max(0, int(audio_dur // video_dur))

    subprocess.run(
        [
            FFMPEG_BIN, "-y",
            "-stream_loop", str(loops_needed),
            "-i", str(video_path),
            "-i", str(audio_path),
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264", "-preset", "medium", "-crf", "22",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k",
            "-shortest",
            "-movflags", "+faststart",
            str(output_path),
        ],
        check=True, capture_output=True,
    )
    print(f"[mux] 저장: {output_path.name}")
    return output_path
