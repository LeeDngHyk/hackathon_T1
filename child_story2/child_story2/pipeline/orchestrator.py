"""
전체 파이프라인 오케스트레이터.

Replicate 크레딧 $5 미만일 때는 rate limit(분당 6, burst 1) 때문에
씬을 병렬로 돌리면 서로 부딪힘. → Phase 1/2의 씬 처리를 순차로 변경.
씬 내부의 이미지·오디오는 여전히 병렬 (TTS는 Typecast라 별개).
"""
from __future__ import annotations

import asyncio
import json
import pickle
from pathlib import Path
from typing import Any

from pipeline.config import (
    BRANCHES_DIR,
    MAX_PARALLEL_JOBS,
    OUTPUT_DIR,
    SCENES_DIR,
    TMP_DIR,
)
from pipeline.enrich_prompts import enrich
from pipeline.image_gen import generate_image
from pipeline.mux import mux_scene
from pipeline.quiz_match import build_quiz_index
from pipeline.story_gen import generate_story
from pipeline.tts_gen import synth_quiz_prompt, synth_scene_audio
from pipeline.video_gen import generate_video


async def build_scene(
    scene: dict[str, Any],
    final_dir: Path,
    sem_image: asyncio.Semaphore,
    sem_audio: asyncio.Semaphore,
    sem_video: asyncio.Semaphore,
) -> Path:
    sid = scene["scene_id"]
    image_path = TMP_DIR / f"{sid}.png"
    audio_path = TMP_DIR / f"{sid}.mp3"
    silent_video_path = TMP_DIR / f"{sid}.silent.mp4"
    final_path = final_dir / f"{sid}.mp4"

    if final_path.exists():
        print(f"[orch] {sid} 이미 완성됨, skip")
        return final_path

    async def gen_image():
        async with sem_image:
            await generate_image(
                scene["image_prompt_en"],
                image_path,
                characters_present=scene.get("characters_present", []),
            )

    async def gen_audio():
        async with sem_audio:
            await synth_scene_audio(scene, audio_path)

    # 이미지·오디오는 벤더가 달라서(Replicate vs Typecast) 병렬 OK
    await asyncio.gather(gen_image(), gen_audio())

    async with sem_video:
        await generate_video(image_path, scene["motion_prompt_en"], silent_video_path)

    mux_scene(silent_video_path, audio_path, final_path)
    return final_path


async def build_all(story: dict[str, Any]) -> dict[str, Any]:
    sem_image = asyncio.Semaphore(MAX_PARALLEL_JOBS)
    sem_audio = asyncio.Semaphore(MAX_PARALLEL_JOBS)
    sem_video = asyncio.Semaphore(1)

    main_scenes = story.get("mainline_scenes", [])
    branch_scenes = story.get("branch_scenes", [])
    preferred = [b for b in branch_scenes if b.get("is_preferred_branch")]
    others = [b for b in branch_scenes if not b.get("is_preferred_branch")]

    # Phase 1: 메인 + 정답 분기, 순차 처리
    print(f"\n[orch] Phase 1: 메인 {len(main_scenes)} + 정답 분기 {len(preferred)} (순차)")
    for i, s in enumerate(main_scenes, 1):
        print(f"[orch] ({i}/{len(main_scenes)}) 메인 씬 {s['scene_id']} 처리 시작")
        await build_scene(s, SCENES_DIR, sem_image, sem_audio, sem_video)
    for i, s in enumerate(preferred, 1):
        print(f"[orch] ({i}/{len(preferred)}) 정답 분기 {s['scene_id']} 처리 시작")
        await build_scene(s, BRANCHES_DIR, sem_image, sem_audio, sem_video)

    # Phase 2: 오답 분기, 순차
    print(f"\n[orch] Phase 2: 오답 분기 {len(others)} (순차)")
    for i, s in enumerate(others, 1):
        print(f"[orch] ({i}/{len(others)}) 오답 분기 {s['scene_id']} 처리 시작")
        await build_scene(s, BRANCHES_DIR, sem_image, sem_audio, sem_video)

    # Phase 3: 퀴즈 오디오
    print(f"\n[orch] Phase 3: 퀴즈 오디오")
    quiz_audio_dir = OUTPUT_DIR / "quizzes"
    quiz_audio_dir.mkdir(exist_ok=True)
    for q in story.get("quizzes", []):
        out = quiz_audio_dir / f"{q['quiz_id']}_prompt.mp3"
        await synth_quiz_prompt(q, out)

    # Phase 4: 퀴즈 임베딩 인덱스
    print(f"\n[orch] Phase 4: 퀴즈 임베딩 인덱스 사전계산")
    build_and_save_quiz_indices(story)

    return build_manifest(story)


def build_and_save_quiz_indices(story: dict[str, Any]) -> None:
    indices_dir = OUTPUT_DIR / "quiz_indices"
    indices_dir.mkdir(exist_ok=True)
    for q in story.get("quizzes", []):
        idx = build_quiz_index(q)
        with (indices_dir / f"{q['quiz_id']}.pkl").open("wb") as f:
            pickle.dump(idx, f)
        print(f"[orch] 퀴즈 인덱스 저장: {q['quiz_id']}.pkl")


def build_manifest(story: dict[str, Any]) -> dict[str, Any]:
    meta = story.get("story_metadata", {})
    main_scenes = story.get("mainline_scenes", [])
    quizzes = {q["appears_after_scene_id"]: q for q in story.get("quizzes", [])}

    seq: list[dict[str, Any]] = []
    for i, s in enumerate(main_scenes):
        sid = s["scene_id"]
        next_sid = s.get("next_scene_id") or (
            main_scenes[i + 1]["scene_id"] if i + 1 < len(main_scenes) else None
        )
        seq.append({"type": "video", "scene_id": sid, "src": f"scenes/{sid}.mp4", "next": next_sid})

        if s.get("interaction", {}).get("has_quiz_after_scene") and sid in quizzes:
            q = quizzes[sid]
            seq.append({
                "type": "quiz",
                "quiz_id": q["quiz_id"],
                "question": q["question"],
                "prompt_audio": f"quizzes/{q['quiz_id']}_prompt.mp3",
                "match_index": f"quiz_indices/{q['quiz_id']}.pkl",
                "merge_to": q["merge_scene_id"],
                "choices": [
                    {
                        "option_id": c["option_id"],
                        "display_text": c["display_text"],
                        "option_tts": c.get("option_tts", c["display_text"]),
                        "semantic_anchor": c.get("semantic_anchor", ""),
                        "accepted_utterances": c.get("accepted_utterances", []),
                        "excluded_meanings": c.get("excluded_meanings", []),
                        "is_preferred_choice": bool(c.get("is_preferred_choice")),
                        "branch_src": f"branches/{c['branch_scene_id']}.mp4",
                        "branch_scene_id": c["branch_scene_id"],
                    }
                    for c in q["choices"]
                ],
            })

    manifest = {
        "story_id": meta.get("story_id", ""),
        "title": meta.get("title", ""),
        "core_message": meta.get("core_message", ""),
        "sequence": seq,
        "ending": story.get("ending", {}),
    }
    path = OUTPUT_DIR / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[orch] 매니페스트 저장: {path}")
    return manifest


async def run_from_parent_input(
    child_profile: dict[str, Any],
    parent_diary: str,
    parent_goal: str,
    work_dir: Path,
) -> None:
    story_path = work_dir / "output_story.json"
    enriched_path = work_dir / "enriched_story.json"

    if story_path.exists():
        print(f"[orch] 스토리 캐시 사용: {story_path.name}")
        story = json.loads(story_path.read_text(encoding="utf-8"))
    else:
        print("[orch] 스토리 생성 시작…")
        story = generate_story(child_profile, parent_diary, parent_goal)
        story_path.write_text(json.dumps(story, ensure_ascii=False, indent=2), encoding="utf-8")

    if enriched_path.exists():
        print(f"[orch] enriched 캐시 사용: {enriched_path.name}")
        story = json.loads(enriched_path.read_text(encoding="utf-8"))
    else:
        story = enrich(story)
        enriched_path.write_text(json.dumps(story, ensure_ascii=False, indent=2), encoding="utf-8")

    await build_all(story)
    print("\n✅ 파이프라인 완료.")


async def run(story_json_path: Path) -> None:
    story = json.loads(story_json_path.read_text(encoding="utf-8"))
    enriched_path = story_json_path.with_name("enriched_story.json")
    if enriched_path.exists():
        story = json.loads(enriched_path.read_text(encoding="utf-8"))
    else:
        story = enrich(story)
        enriched_path.write_text(json.dumps(story, ensure_ascii=False, indent=2), encoding="utf-8")
    await build_all(story)