/**
 * 동화 재생기.
 * - manifest.json을 로드해 sequence를 순차 재생
 * - 퀴즈 씬에서 4지선다 UI 표시
 * - 아이가 선택(터치 또는 음성) → 해당 branch_src 재생 → merge_to 씬으로 복귀
 *
 * 음성 매칭은 브라우저 SpeechRecognition으로 STT 받은 뒤,
 * 선택지의 display_text/option_tts/accepted_utterances 문자열들과
 * 간단한 유사도(문자열 포함 우선 → 단어 겹침) 계산.
 * 프로덕션에서는 서버측 임베딩 매칭으로 대체 권장.
 */

const $ = (id) => document.getElementById(id);
const screens = {
  start: $("start-screen"),
  play: $("play-screen"),
  quiz: $("quiz-screen"),
  end: $("end-screen"),
};

let manifest = null;
let cursor = 0;        // sequence 인덱스
const video = $("video");

// ─────────────────────────────────────────────────────────────
// 화면 전환
// ─────────────────────────────────────────────────────────────
function showScreen(name) {
  Object.values(screens).forEach((s) => s.classList.remove("active"));
  screens[name].classList.add("active");
}

// ─────────────────────────────────────────────────────────────
// 로드
// ─────────────────────────────────────────────────────────────
async function loadManifest() {
  const resp = await fetch("../output/manifest.json");
  if (!resp.ok) throw new Error("manifest.json 로드 실패");
  manifest = await resp.json();

  $("story-title").textContent = manifest.title || "동화";
  $("story-message").textContent = manifest.core_message || "";
}

// ─────────────────────────────────────────────────────────────
// 재생 루프
// ─────────────────────────────────────────────────────────────
function playNext() {
  if (cursor >= manifest.sequence.length) {
    showEnding();
    return;
  }

  const item = manifest.sequence[cursor];
  cursor += 1;

  if (item.type === "video") {
    playVideo(`../output/${item.src}`, () => playNext());
  } else if (item.type === "quiz") {
    showQuiz(item);
  }
}

function playVideo(src, onEnded) {
  showScreen("play");
  video.src = src;
  video.onended = onEnded;
  video.play().catch((err) => {
    console.warn("자동재생 실패, 사용자 제스처 필요:", err);
  });
}

// ─────────────────────────────────────────────────────────────
// 퀴즈
// ─────────────────────────────────────────────────────────────
let currentQuiz = null;

function showQuiz(quiz) {
  currentQuiz = quiz;
  showScreen("quiz");
  $("quiz-question").textContent = quiz.question;

  const container = $("quiz-choices");
  container.innerHTML = "";
  quiz.choices.forEach((c) => {
    const btn = document.createElement("button");
    btn.className = "choice";
    btn.dataset.optionId = c.option_id;
    btn.innerHTML = `<span class="option-label">${c.option_id}</span><span>${c.display_text}</span>`;
    btn.addEventListener("click", () => onChoiceSelected(c));
    container.appendChild(btn);
  });

  // 질문 오디오 자동 재생
  if (quiz.prompt_audio) {
    const audio = new Audio(`../output/${quiz.prompt_audio}`);
    audio.play().catch(() => {});
  }

  $("quiz-status").textContent = "";
}

function onChoiceSelected(choice) {
  // 선택 하이라이트
  document.querySelectorAll(".choice").forEach((el) => {
    el.classList.toggle("selected", el.dataset.optionId === choice.option_id);
  });

  // 분기 영상 재생 → merge_to 씬으로 커서 이동 → 다음 재생
  const branchSrc = `../output/${choice.branch_src}`;
  const mergeTo = currentQuiz.merge_to;

  setTimeout(() => {
    playVideo(branchSrc, () => {
      // sequence에서 merge_to에 해당하는 인덱스 찾아 커서 세팅
      const mergeIdx = manifest.sequence.findIndex(
        (it) => it.type === "video" && it.scene_id === mergeTo
      );
      if (mergeIdx !== -1) cursor = mergeIdx;
      playNext();
    });
  }, 400);
}

// ─────────────────────────────────────────────────────────────
// 음성 입력 (SpeechRecognition + 간이 매칭)
// ─────────────────────────────────────────────────────────────
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
let recognition = null;

if (SpeechRecognition) {
  recognition = new SpeechRecognition();
  recognition.lang = "ko-KR";
  recognition.interimResults = false;
  recognition.maxAlternatives = 3;

  recognition.onresult = (event) => {
    const alternatives = Array.from(event.results[0]).map((r) => r.transcript);
    const utterance = alternatives[0];
    $("quiz-status").textContent = `들었어요: "${utterance}"`;
    const matched = matchChoice(alternatives);
    if (matched) {
      onChoiceSelected(matched);
    } else {
      $("quiz-status").textContent = `"${utterance}" — 다시 한 번 말해줄래요?`;
    }
  };
  recognition.onerror = () => {
    $("quiz-status").textContent = "다시 시도해주세요.";
    $("mic-btn").classList.remove("recording");
  };
  recognition.onend = () => {
    $("mic-btn").classList.remove("recording");
  };
}

$("mic-btn").addEventListener("click", () => {
  if (!recognition) {
    $("quiz-status").textContent = "이 브라우저는 음성 입력을 지원하지 않아요. 손가락으로 눌러주세요.";
    return;
  }
  $("mic-btn").classList.add("recording");
  $("quiz-status").textContent = "듣고 있어요…";
  recognition.start();
});

/**
 * 간이 매칭:
 * 1) 후보 발화(alternatives) 중 하나라도 선택지의 accepted_utterances에
 *    포함되거나 포함하는 관계면 우선 매칭.
 * 2) 없으면 단어 겹침 개수로 최고 점수 선택지 선택. 최소 임계 미달이면 null.
 *
 * 프로덕션: 서버측 임베딩 매칭으로 대체. semantic_anchor와 accepted_utterances,
 * excluded_meanings를 사전 벡터화하고 코사인 유사도로 판정.
 */
function matchChoice(alternatives) {
  const cleaned = alternatives.map(normalize);

  // 1) 정확·부분 일치
  for (const alt of cleaned) {
    for (const c of currentQuiz.choices) {
      const anchors = [
        c.display_text,
        c.option_tts,
        ...(c.accepted_utterances || []),
      ].map(normalize);
      if (anchors.some((a) => alt.includes(a) || a.includes(alt))) {
        return c;
      }
    }
  }

  // 2) 단어 겹침 스코어
  let best = null;
  let bestScore = 0;
  for (const c of currentQuiz.choices) {
    const anchorTokens = new Set(
      [c.display_text, ...(c.accepted_utterances || [])]
        .flatMap((t) => normalize(t).split(/\s+/))
    );
    for (const alt of cleaned) {
      const altTokens = alt.split(/\s+/);
      const overlap = altTokens.filter((t) => anchorTokens.has(t)).length;
      if (overlap > bestScore) {
        best = c;
        bestScore = overlap;
      }
    }
  }
  return bestScore >= 1 ? best : null;
}

function normalize(s) {
  return (s || "").toLowerCase().replace(/[^\p{L}\p{N}\s]/gu, "").trim();
}

// ─────────────────────────────────────────────────────────────
// 결말
// ─────────────────────────────────────────────────────────────
function showEnding() {
  const e = manifest.ending || {};
  $("closing-message").textContent = e.closing_message || "";
  $("follow-up").textContent = e.parent_follow_up_question || "";
  $("practice").textContent = e.offline_practice_suggestion || "";
  showScreen("end");
}

$("replay-btn").addEventListener("click", () => {
  cursor = 0;
  showScreen("start");
});

// ─────────────────────────────────────────────────────────────
// 시작 버튼
// ─────────────────────────────────────────────────────────────
$("start-btn").addEventListener("click", () => {
  cursor = 0;
  playNext();
});

// ─────────────────────────────────────────────────────────────
// 부팅
// ─────────────────────────────────────────────────────────────
loadManifest().catch((err) => {
  $("story-title").textContent = "동화를 불러올 수 없어요";
  $("story-message").textContent = err.message;
  $("start-btn").disabled = true;
});
