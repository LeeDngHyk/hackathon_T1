"""
manifest.json 을 따라 씬을 하나의 mp4 로 이어붙임.

사용법:
    python concat_video.py                    # 정답 분기(preferred) 경로 → full_story.mp4
    python concat_video.py --option A         # A 선택지 경로 → full_story_A.mp4
    python concat_video.py --all-branches     # 4개 선택지 각각 → full_story_A/B/C/D.mp4
    python concat_video.py --fast             # 재인코딩 없이 (모든 클립 스펙 완전히 같아야 함)
    python concat_video.py --check            # 클립별 오디오 유무 진단만 (concat 안 함)

구현:
    기본 모드는 ffmpeg concat filter 를 사용 — 오디오·비디오 스트림을 명시적으로
    매핑해서 확실히 포함되도록 함. concat demuxer 는 재인코딩 시 오디오를
    조용히 드롭하는 경우가 있어서 지양.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from pipeline.config import FFMPEG_BIN, FFPROBE_BIN, OUTPUT_DIR


# ─────────────────────────────────────────────────────────────
# manifest & 클립 목록
# ─────────────────────────────────────────────────────────────
def load_manifest() -> dict:
    p = OUTPUT_DIR / "manifest.json"
    if not p.exists():
        print(f"[concat] manifest.json 이 없어요: {p}")
        sys.exit(1)
    return json.loads(p.read_text(encoding="utf-8"))


def build_clip_list(manifest: dict, target_option_id: str = "preferred") -> list[Path]:
    clips: list[Path] = []
    for item in manifest["sequence"]:
        if item["type"] == "video":
            clips.append(OUTPUT_DIR / item["src"])
        elif item["type"] == "quiz":
            if target_option_id == "preferred":
                choice = next(
                    (c for c in item["choices"] if c.get("is_preferred_choice")),
                    item["choices"][0],
                )
            else:
                choice = next(
                    (c for c in item["choices"] if c["option_id"] == target_option_id),
                    None,
                )
                if not choice:
                    raise ValueError(
                        f"퀴즈 '{item['quiz_id']}' 에 option_id='{target_option_id}' 없음"
                    )
            clips.append(OUTPUT_DIR / choice["branch_src"])
    return clips


# ─────────────────────────────────────────────────────────────
# 진단: 각 클립의 오디오 유무 확인
# ─────────────────────────────────────────────────────────────
def probe_streams(clip: Path) -> dict:
    """ffprobe 로 비디오·오디오 스트림 유무 확인."""
    r = subprocess.run(
        [
            FFPROBE_BIN, "-v", "error", "-show_streams",
            "-of", "json", str(clip),
        ],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(r.stdout)
    has_video = any(s.get("codec_type") == "video" for s in data.get("streams", []))
    has_audio = any(s.get("codec_type") == "audio" for s in data.get("streams", []))
    return {"has_video": has_video, "has_audio": has_audio}


def check_clips(clips: list[Path]) -> bool:
    print("[check] 클립별 스트림 진단:")
    all_ok = True
    for c in clips:
        if not c.exists():
            print(f"  ✗ {c.name}: 파일 없음")
            all_ok = False
            continue
        info = probe_streams(c)
        v = "V" if info["has_video"] else "-"
        a = "A" if info["has_audio"] else "-"
        ok = info["has_video"] and info["has_audio"]
        mark = "✓" if ok else "✗"
        print(f"  {mark} {c.name} [{v}{a}]")
        if not ok:
            all_ok = False
    return all_ok


# ─────────────────────────────────────────────────────────────
# ffmpeg concat
# ─────────────────────────────────────────────────────────────
def concat_via_filter(clips: list[Path], output_path: Path) -> None:
    """
    concat filter 방식 — 스트림별 명시 매핑으로 오디오 확실히 포함.
    각 클립의 v:0 과 a:0 을 명시적으로 지정.
    """
    cmd = [FFMPEG_BIN, "-y"]
    for c in clips:
        cmd.extend(["-i", str(c.resolve())])

    n = len(clips)
    # [0:v:0][0:a:0][1:v:0][1:a:0]...concat=n=N:v=1:a=1[outv][outa]
    parts = "".join(f"[{i}:v:0][{i}:a:0]" for i in range(n))
    filter_str = f"{parts}concat=n={n}:v=1:a=1[outv][outa]"

    cmd.extend([
        "-filter_complex", filter_str,
        "-map", "[outv]",
        "-map", "[outa]",
        "-c:v", "libx264", "-preset", "medium", "-crf", "22",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        str(output_path),
    ])

    print(f"[concat] filter 방식으로 {n}개 클립 합치는 중…")
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print("[concat] ffmpeg 실패:")
        print(e.stderr.decode("utf-8", errors="ignore")[-2000:])
        raise


def concat_via_demuxer_copy(clips: list[Path], output_path: Path) -> None:
    """빠른 concat — 재인코딩 없이 스트림 복사. 클립 스펙이 완전 동일해야 함."""
    list_file = output_path.with_suffix(".concat.txt")
    list_file.write_text(
        "\n".join(f"file '{c.resolve().as_posix()}'" for c in clips),
        encoding="utf-8",
    )
    cmd = [
        FFMPEG_BIN, "-y", "-f", "concat", "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",
        "-movflags", "+faststart",
        str(output_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print("[concat] ffmpeg 실패:")
        print(e.stderr.decode("utf-8", errors="ignore")[-2000:])
        raise
    finally:
        list_file.unlink(missing_ok=True)


def concat_clips(clips: list[Path], output_path: Path, fast: bool = False) -> None:
    missing = [c for c in clips if not c.exists()]
    if missing:
        print("[concat] 다음 파일이 없어요:")
        for m in missing:
            print(f"  - {m}")
        sys.exit(1)

    print(f"[concat] {len(clips)}개 클립 → {output_path.name}")
    for c in clips:
        print(f"         · {c.name}")

    if fast:
        concat_via_demuxer_copy(clips, output_path)
    else:
        concat_via_filter(clips, output_path)

    # 결과 검증
    info = probe_streams(output_path)
    v = "✓" if info["has_video"] else "✗"
    a = "✓" if info["has_audio"] else "✗"
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"[concat] 완료: {output_path} ({size_mb:.1f} MB)")
    print(f"         video: {v}  audio: {a}")
    if not info["has_audio"]:
        print("[concat] ⚠ 오디오가 없습니다. 원본 클립을 --check 로 확인해보세요.")


# ─────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(description="manifest 대로 씬을 하나의 mp4 로 합치기")
    parser.add_argument("--option", default="preferred",
                        help="preferred | A | B | C | D (기본: preferred)")
    parser.add_argument("--all-branches", action="store_true",
                        help="4개 선택지 각각의 완주본을 모두 생성")
    parser.add_argument("--fast", action="store_true",
                        help="재인코딩 없이 concat (모든 클립 스펙이 완전 동일할 때만)")
    parser.add_argument("--check", action="store_true",
                        help="클립별 스트림 진단만. concat 안 함")
    args = parser.parse_args()

    manifest = load_manifest()

    if args.check:
        clips = build_clip_list(manifest, target_option_id="preferred")
        ok = check_clips(clips)
        return 0 if ok else 1

    if args.all_branches:
        first_quiz = next((i for i in manifest["sequence"] if i["type"] == "quiz"), None)
        if not first_quiz:
            print("[concat] 퀴즈 없음, 하나만 생성")
            clips = build_clip_list(manifest, "preferred")
            concat_clips(clips, OUTPUT_DIR / "full_story.mp4", fast=args.fast)
            return 0
        for choice in first_quiz["choices"]:
            oid = choice["option_id"]
            clips = build_clip_list(manifest, target_option_id=oid)
            out = OUTPUT_DIR / f"full_story_{oid}.mp4"
            concat_clips(clips, out, fast=args.fast)
    else:
        target = args.option
        clips = build_clip_list(manifest, target_option_id=target)
        suffix = "" if target == "preferred" else f"_{target}"
        out = OUTPUT_DIR / f"full_story{suffix}.mp4"
        concat_clips(clips, out, fast=args.fast)

    return 0


if __name__ == "__main__":
    sys.exit(main())