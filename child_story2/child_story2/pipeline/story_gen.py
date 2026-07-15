"""
부모 입력 → 구조화된 동화 스토리 JSON.

입력:
  - child_profile: 아이 정보 (이름, 나이, 성별, 특징) — 기본값에서 로드
  - parent_diary: 부모의 육아일기 (자유 텍스트)
  - parent_goal: 부모가 훈육하고 싶은 지향점 (예: "블록 정리를 스스로 하기")
  - character_visual_lock: 캐릭터 시각 프리셋 (config.CHARACTER_VISUAL_LOCK)
  - defaults: 씬 수, 톤, 그림체 등 잠긴 기본값

출력: orchestrator.build_manifest()가 그대로 소비할 수 있는 스토리 JSON.
    schema:
      story_metadata: {story_id, title, core_message, age_range}
      mainline_scenes: [{scene_id, location, time_of_day, characters_present,
                        story_event, child_emotion, narration,
                        dialogue: [{speaker_id, text, voice_direction}],
                        interaction: {has_quiz_after_scene: bool},
                        next_scene_id}]
      branch_scenes:  [ ... + is_preferred_branch, merge_scene_id ]
      quizzes: [{quiz_id, appears_after_scene_id, merge_scene_id,
                 question, voice_prompt,
                 choices: [{option_id, display_text, option_tts,
                            semantic_anchor, accepted_utterances,
                            excluded_meanings, is_preferred_choice,
                            branch_scene_id}]}]
      ending: {narration, moral, parent_note}

이후 pipeline.enrich_prompts.enrich() 가 image_prompt_en, motion_prompt_en
을 부착해 최종 enriched_story.json 을 만든다.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from anthropic import Anthropic

from pipeline.config import (
    ANTHROPIC_API_KEY,
    CHARACTER_VISUAL_LOCK,
    STORY_MODEL,
    STORY_MAX_TOKENS,
    STORY_DEFAULTS,
)


# ═════════════════════════════════════════════════════════════
# 시스템 프롬프트 — 파이프라인의 계약(schema + 톤 + 안전) 잠금
# 이 프롬프트는 다음을 모두 강제한다:
#   1) 정확한 JSON 스키마 (파이프라인 하류가 소비 가능)
#   2) 씬 수·톤·연령대 고정 (기본값에서 주입)
#   3) 훈육 목표를 스토리 갈등의 핵심축으로 삼기
#   4) 4지선다 퀴즈의 음성 매칭용 메타(semantic_anchor 등) 채우기
#   5) 유아 안전(공포·폭력·훈계조·낙인 없음)
# ═════════════════════════════════════════════════════════════
def build_system_prompt(
    defaults: dict[str, Any],
    character_visual_lock: dict[str, str],
) -> str:
    speaker_ids = list(character_visual_lock.keys())
    return f"""당신은 3~6세 유아 대상 훈육 동화의 시나리오 작가입니다.
부모가 준 육아일기와 훈육 목표를 바탕으로, 아이가 스스로 깨닫도록
설계된 인터랙티브 동화의 스토리 JSON을 생성합니다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[절대 규칙 — 위반 시 파이프라인이 깨집니다]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 출력은 **오직 하나의 유효한 JSON 오브젝트**. 마크다운·코드펜스·설명 없음.
2. 스키마는 아래 [출력 스키마]를 문자 그대로 따를 것. 필드 누락·추가 금지.
3. 등장인물의 speaker_id는 오직 다음 값들만 사용:
   {json.dumps(speaker_ids, ensure_ascii=False)}
   (예: "narrator", "protagonist", "obj_1", "sibling")
4. 모든 대사·내레이션은 **한국어**. 3~6세가 알아듣는 어휘와 짧은 문장.
5. 씬 ID는 메인 "S01"부터 순번, 분기 "Q01_A"/"Q01_B"/"Q01_C"/"Q01_D" 형식.
6. 퀴즈 choices는 **정확히 4개**. option_id는 "A", "B", "C", "D".

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[고정 기본값 — 이 값들을 반드시 준수]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- 메인 씬 개수: {defaults['scene_count']}개 (S01 ~ S{defaults['scene_count']:02d})
- 퀴즈 개수: {defaults['quiz_count']}개
- 각 퀴즈의 분기 씬: 4개 (선택지당 하나, 정답 분기 1 + 오답 분기 3)
- 스토리 톤: {defaults['tone']}
- 대상 연령: {defaults['age_range']}
- 그림체(참고): {defaults['art_style_note']}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[스토리 설계 원칙]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

원칙 1: **훈육 목표가 갈등의 핵심축**
- 부모가 준 parent_goal이 이야기의 중심 갈등이 되도록 설계.
- 나쁜 행동을 직접 지적하지 말고, 상황을 통해 **아이가 스스로 알아채도록** 유도.

원칙 2: **훈계 금지, 공감 우선**
- "그러면 안 돼", "착한 아이는 ~해요" 같은 훈계조 대사 금지.
- 조력자 캐릭터(예: obj_1 애착인형)는 지시하지 않고 **질문·감탄·궁금해하기**만.

원칙 3: **부모의 육아일기에서 세부를 반드시 반영**
- 일기에 언급된 장소·물건·에피소드를 한 개 이상 씬에 실물로 등장시킴.
- 이것이 "우리 아이 이야기"라는 실감을 만드는 핵심.

원칙 4: **퀴즈는 훈육의 결정 포인트**
- 퀴즈는 아이가 "다음에 어떻게 할까?"를 고르는 순간에 배치.
- 4개 선택지 중 하나는 parent_goal에 부합하는 **정답 분기(is_preferred_choice=true)**.
- 나머지 3개는 유아가 실제로 할 법한 그럴듯한 오답들.
- 어떤 선택도 "틀렸다"고 낙인찍지 말 것. 오답 분기에서는 결과를 겪고 자연스럽게 배움.

원칙 5: **모든 분기는 merge_scene_id에서 합류**
- 분기 씬은 각 5초 안팎의 짧은 결과를 보여준 뒤 다시 본류로 돌아옴.
- 정답 분기: 자연스러운 성취감, 조력자의 다정한 반응.
- 오답 분기: 결과를 겪음 → 조력자가 감정을 짚어줌 → 다시 시도할 여지.

원칙 6: **음성 매칭용 메타 필수**
- 각 choice에 다음 세 필드를 반드시 채움 (아이 음성 → 선택지 매칭에 사용):
  · semantic_anchor: 그 선택의 **의미를 요약한 한 문장** (임베딩 기준).
      예: "블록을 상자에 담아 정리한다"
  · accepted_utterances: 아이가 실제로 말할 법한 표현들 3~6개.
      예: ["정리할래", "상자에 담을래", "치울래요", "정리해요", "다 담아요"]
  · excluded_meanings: 이 선택과 헷갈리면 안 되는 정반대 의미 2~4개.
      예: ["그냥 둘래", "안 치울래", "나중에 할래", "엄마가 해줘"]

원칙 7: **voice_direction은 감정 힌트만**
- Typecast TTS의 감정 매핑을 위해 각 대사에 짧은 한국어 힌트를 붙임.
  가능한 힌트 키워드: "차분하게", "밝고 신나게", "울먹이며", "부드럽고 다정하게",
  "속삭이듯", "활기차고 또박또박", "조금 속상해하며".
- 문학적 지문 금지. TTS는 감정 키워드만 참고함.

원칙 8: **안전**
- 무섭거나 어두운 장면 금지 (밤 혼자, 사라짐, 다침, 벌 등).
- 신체·외모·성격에 대한 부정적 언급 금지.
- 실존 인물·브랜드·저작권 캐릭터 언급 금지.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[스토리 구조 템플릿 — {defaults['scene_count']}씬 기준]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

S01: 도입 — 아이의 일상, 조력자 등장, 오늘의 배경 소개
S02: 발단 — parent_goal과 관련된 상황이 자연스럽게 발생
S03: 갈등 — 아이가 선택의 기로에 놓임 → **여기서 퀴즈 Q01**
      (S03 뒤에 Q01, 선택에 따라 Q01_A/B/C/D → S04에서 합류)
S04: 결과 — 어떤 분기를 골랐든 이 씬으로 합류. 배움의 순간.
S05: 성장 — 아이가 스스로 다시 시도하거나 조력자와 성찰
S06: 마무리 — 따뜻한 결말, 하루의 소감

(quiz_count가 2 이상이면 S04 뒤에 Q02, S05에서 합류하는 식으로 확장)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[출력 스키마 — 이 형태 그대로]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{{
  "story_metadata": {{
    "story_id": "짧은 slug (영문 소문자·하이픈)",
    "title": "동화 제목 (한국어)",
    "core_message": "아이에게 전하고 싶은 한 문장 메시지",
    "age_range": "{defaults['age_range']}"
  }},
  "mainline_scenes": [
    {{
      "scene_id": "S01",
      "location": "장소 (예: 지우의 방)",
      "time_of_day": "시간대 (아침/낮/저녁)",
      "characters_present": ["protagonist", "obj_1"],
      "story_event": "이 씬에서 일어나는 일을 한국어 1~2문장으로",
      "child_emotion": "이 씬에서 아이가 느끼는 감정",
      "narration": "내레이터가 읽어줄 한국어 문장 (1~3문장, 짧게)",
      "dialogue": [
        {{
          "speaker_id": "protagonist",
          "text": "대사 한 줄",
          "voice_direction": "밝고 신나게"
        }}
      ],
      "interaction": {{"has_quiz_after_scene": false}},
      "next_scene_id": "S02"
    }}
  ],
  "branch_scenes": [
    {{
      "scene_id": "Q01_A",
      "location": "...", "time_of_day": "...",
      "characters_present": ["..."],
      "story_event": "...", "child_emotion": "...",
      "narration": "...",
      "dialogue": [...],
      "is_preferred_branch": true,
      "merge_scene_id": "S04"
    }}
  ],
  "quizzes": [
    {{
      "quiz_id": "Q01",
      "appears_after_scene_id": "S03",
      "merge_scene_id": "S04",
      "question": "아이에게 화면에 보여줄 질문 (한국어, 짧고 명확)",
      "voice_prompt": "조력자가 다정하게 소리내어 묻는 문장 (한국어)",
      "choices": [
        {{
          "option_id": "A",
          "display_text": "선택지 텍스트 (화면 표시용, 짧게)",
          "option_tts": "선택지를 음성으로 안내할 때의 문장",
          "semantic_anchor": "이 선택의 의미를 요약한 한 문장",
          "accepted_utterances": ["아이가 말할 법한 표현1", "...3~6개"],
          "excluded_meanings": ["정반대 의미의 표현1", "...2~4개"],
          "is_preferred_choice": true,
          "branch_scene_id": "Q01_A"
        }}
      ]
    }}
  ],
  "ending": {{
    "narration": "이야기를 마무리하는 내레이터 문장",
    "moral": "아이가 배운 것을 한 문장으로",
    "parent_note": "부모에게 전하는 한 줄 코멘트 (앱 안에서 부모용 카드에 표시)"
  }}
}}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[자체 점검 — JSON을 내보내기 직전에 확인]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

□ mainline_scenes 길이 == {defaults['scene_count']}
□ quizzes 길이 == {defaults['quiz_count']}
□ 각 quiz의 choices 길이 == 4
□ 각 quiz에서 is_preferred_choice=true 인 choice는 정확히 1개
□ 모든 branch_scene_id가 branch_scenes 안에 실존
□ 모든 merge_scene_id가 mainline_scenes 안에 실존
□ 모든 speaker_id가 허용 목록 안에 있음
□ 훈계조 문장이 없는지, 공포·낙인 요소가 없는지
□ 육아일기의 세부(장소·물건·에피소드)가 최소 1개 씬에 반영됐는지
"""


# ═════════════════════════════════════════════════════════════
# 유저 프롬프트 — 부모가 입력한 실제 데이터
# ═════════════════════════════════════════════════════════════
def build_user_prompt(
    child_profile: dict[str, Any],
    parent_diary: str,
    parent_goal: str,
) -> str:
    return f"""아래 정보로 동화 스토리 JSON을 생성해 주세요.

[아이 프로필]
{json.dumps(child_profile, ensure_ascii=False, indent=2)}

[부모의 육아일기]
{parent_diary.strip()}

[부모가 원하는 훈육 지향점]
{parent_goal.strip()}

위 정보를 반영해, 시스템 프롬프트의 규칙과 스키마를 정확히 지켜 JSON만 출력하세요."""


# ═════════════════════════════════════════════════════════════
# 유틸
# ═════════════════════════════════════════════════════════════
def strip_code_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        first_nl = t.find("\n")
        if first_nl != -1:
            t = t[first_nl + 1 :]
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip()


def validate_story(story: dict[str, Any], defaults: dict[str, Any]) -> list[str]:
    """스토리 JSON의 구조적 무결성을 검증. 에러 리스트 반환 (빈 리스트=OK)."""
    errors: list[str] = []
    mains = story.get("mainline_scenes", [])
    branches = story.get("branch_scenes", [])
    quizzes = story.get("quizzes", [])

    if len(mains) != defaults["scene_count"]:
        errors.append(f"mainline_scenes 개수 {len(mains)} != {defaults['scene_count']}")
    if len(quizzes) != defaults["quiz_count"]:
        errors.append(f"quizzes 개수 {len(quizzes)} != {defaults['quiz_count']}")

    main_ids = {s["scene_id"] for s in mains}
    branch_ids = {s["scene_id"] for s in branches}

    for q in quizzes:
        qid = q.get("quiz_id", "?")
        choices = q.get("choices", [])
        if len(choices) != 4:
            errors.append(f"{qid}: choices 4개 아님 ({len(choices)}개)")
        preferred = [c for c in choices if c.get("is_preferred_choice")]
        if len(preferred) != 1:
            errors.append(f"{qid}: is_preferred_choice=true가 1개 아님 ({len(preferred)}개)")
        merge = q.get("merge_scene_id")
        if merge not in main_ids:
            errors.append(f"{qid}: merge_scene_id '{merge}'가 메인에 없음")
        for c in choices:
            bsid = c.get("branch_scene_id")
            if bsid not in branch_ids:
                errors.append(f"{qid}/{c.get('option_id')}: branch_scene_id '{bsid}' 미존재")
            for req in ("semantic_anchor", "accepted_utterances", "excluded_meanings"):
                if not c.get(req):
                    errors.append(f"{qid}/{c.get('option_id')}: {req} 비어있음")

    return errors


# ═════════════════════════════════════════════════════════════
# 메인 진입점
# ═════════════════════════════════════════════════════════════
def generate_story(
    child_profile: dict[str, Any],
    parent_diary: str,
    parent_goal: str,
    defaults: dict[str, Any] | None = None,
    character_visual_lock: dict[str, str] | None = None,
) -> dict[str, Any]:
    """부모 입력 → 스토리 JSON. 검증 실패 시 1회 자동 재시도."""
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY가 필요합니다.")

    defaults = defaults or STORY_DEFAULTS
    character_visual_lock = character_visual_lock or CHARACTER_VISUAL_LOCK

    system = build_system_prompt(defaults, character_visual_lock)
    user = build_user_prompt(child_profile, parent_diary, parent_goal)

    client = Anthropic()

    def _call(extra_user: str = "") -> dict[str, Any]:
        resp = client.messages.create(
            model=STORY_MODEL,
            max_tokens=STORY_MAX_TOKENS,
            system=system,
            messages=[{"role": "user", "content": user + extra_user}],
        )
        raw = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        return json.loads(strip_code_fence(raw))

    story = _call()
    errors = validate_story(story, defaults)
    if errors:
        print("[story_gen] 검증 실패, 1회 재시도:")
        for e in errors:
            print(f"  - {e}")
        feedback = "\n\n[이전 출력의 오류 — 반드시 고쳐서 다시 출력]\n" + "\n".join(f"- {e}" for e in errors)
        story = _call(feedback)
        errors = validate_story(story, defaults)
        if errors:
            raise RuntimeError(f"스토리 검증 실패 (재시도 후에도): {errors}")

    print(f"[story_gen] 생성 완료: {story['story_metadata']['title']}")
    return story


def generate_story_to_file(
    child_profile: dict[str, Any],
    parent_diary: str,
    parent_goal: str,
    output_path: Path,
) -> Path:
    story = generate_story(child_profile, parent_diary, parent_goal)
    output_path.write_text(
        json.dumps(story, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[story_gen] 저장: {output_path}")
    return output_path


if __name__ == "__main__":
    # 스모크 테스트
    demo_profile = {
        "name": "지우",
        "age": 5,
        "gender": "female",
        "traits": ["호기심 많음", "부끄러움 잘 탐", "블록놀이 좋아함"],
    }
    demo_diary = (
        "지우가 요즘 블록으로 성을 만들고 나서 정리를 안 해요. "
        "저녁마다 거실 바닥이 블록 천지라 제가 대신 치우게 돼요. "
        "혼내면 울어버려서 어떻게 알려줄지 고민이에요."
    )
    demo_goal = "블록 놀이 후에 스스로 정리하는 습관을 만들고 싶어요."
    generate_story_to_file(
        demo_profile, demo_diary, demo_goal, Path("output_story.json")
    )
