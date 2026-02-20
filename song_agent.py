#!/usr/bin/env python3
"""
송근용 에이전트 — 하이브리드 분석 엔진
========================================
파인튜닝된 GPT 모델 + 실시간 컨텍스트 주입으로 작동하는
"이 사람이라면 어떻게 분석할까?" 엔진.

아키텍처:
  Layer A — 스타일/추론 (파인튜닝 모델이 담당)
    → 송근용의 사고 패턴, 판단 프레임워크, 분석 루틴
  Layer B — 팩트/시의성 (실시간 주입)
    → 실적 데이터, 가격, 뉴스, 산업 지표 등

사전 준비:
    pip install openai
    set OPENAI_API_KEY=sk-...

사용법:
    python song_agent.py                           # 대화형 모드
    python song_agent.py --question "반도체 사이클 분석해줘"
    python song_agent.py --context "삼성전자 2025Q4 영업이익 6.5조"
    python song_agent.py --mode frame              # 프레임 카드 목록 보기
    python song_agent.py --mode analyze --topic "AI 반도체"  # 자동 분석

모델 지정:
    python song_agent.py --model ft:gpt-4o-mini-2024-07-18:...:song-agent
    python song_agent.py --model gpt-4o-mini       # 파인튜닝 전 프로토타입
"""

import argparse
import json
import os
import sys
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    print("❌ openai 패키지가 필요합니다.")
    print("   설치: pip install openai")
    sys.exit(1)


# ============================================================
# 설정
# ============================================================
STATE_FILE = Path(__file__).parent / "data" / "finetune_state.json"
FRAMES_FILE = Path(__file__).parent / "data" / "song_frames.json"
SYSTEM_PROMPT_FILE = Path(__file__).parent / "data" / "system_prompt_v2.txt"

# 파인튜닝 전 프로토타입용 — 풀 시스템 프롬프트
FALLBACK_SYSTEM_PROMPT = """\
당신은 '송근용 에이전트'입니다.

## 페르소나
투자 블로그 blog.naver.com/tosoha1의 저자 송근용의 사고방식을 내재화한 분석 에이전트.
2005년부터 1,300여 편의 글을 써 온 투자자의 관점으로 시장 현상을 분석합니다.

## 핵심 사고 프레임워크
1. 산업 사이클: 재고→가격→CAPEX→공급→가격 순환의 위치 판별
2. 수요-공급 지형: 수요 증가율 vs 공급 증가율의 격차가 이익의 원천
3. 실적의 질 (Q vs P): 물량(Q) 성장인지 가격(P) 효과인지 구분
4. 컨퍼런스콜 해석: 경영진 발언에서 톤 변화, 숨겨진 시그널 포착
5. 밸류에이션 앵커링: 절대가치보다 "지금 시장이 무엇을 반영하고 있나" 기준
6. 생존 우선: 상방보다 하방 리스크를 먼저 점검
7. 겸손/불확실성 인정: 확신 수준을 명시하고 틀릴 가능성 언급
8. 역사적 유사국면: 과거 비슷한 상황을 찾아 비교

## 답변 구조
1. 현상 정리 (무슨 일이 일어났나)
2. 프레임 적용 (어떤 분석 틀로 볼 것인가)
3. 핵심 체크포인트 (확인해야 할 데이터/이벤트)
4. 시나리오 (긍정/부정/베이스 케이스)
5. 겸손 코멘트 (불확실성, 내가 틀릴 수 있는 지점)

## 규칙
- "이 사람이라면 어떻게 분석할까?"의 관점으로 답변
- 가격 예측이 아니라 현상의 구조적 이해가 목적
- 데이터가 부족하면 "이 부분은 확인이 필요합니다"라고 명시
- 한국어로 자연스럽게, 블로그 글 톤으로 답변
"""


# ============================================================
# 프레임 카드 관리
# ============================================================
def load_frames() -> list:
    """저장된 프레임 카드 로드"""
    if FRAMES_FILE.exists():
        with open(FRAMES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def show_frames():
    """프레임 카드 목록 출력"""
    frames = load_frames()
    if not frames:
        print("  ⚠️ 프레임 카드가 없습니다.")
        print("     먼저 convert_to_finetune_v2.py를 실행하세요.")
        return

    print(f"\n{'='*60}")
    print(f"  📋 송근용 사고 프레임 카드 ({len(frames)}개)")
    print(f"{'='*60}\n")

    for i, f in enumerate(frames, 1):
        print(f"  [{f['id']}] {f['name']}")
        print(f"    {f['description'][:80]}...")
        print(f"    키워드: {', '.join(f['keywords'][:5])}")
        print()


def find_relevant_frames(text: str, top_k: int = 3) -> list:
    """텍스트에서 관련 프레임 카드 자동 선택"""
    frames = load_frames()
    if not frames:
        return []

    scored = []
    text_lower = text.lower()

    for frame in frames:
        score = 0
        # 키워드 매칭
        for kw in frame.get("keywords", []):
            if kw.lower() in text_lower:
                score += 2
        # 이름 매칭
        if frame["name"].replace(" ", "") in text.replace(" ", ""):
            score += 3
        # 설명 키워드 매칭
        desc_words = frame.get("description", "").split()
        for w in desc_words:
            if len(w) > 2 and w in text:
                score += 0.5

        if score > 0:
            scored.append((score, frame))

    scored.sort(key=lambda x: -x[0])
    return [f for _, f in scored[:top_k]]


# ============================================================
# 시스템 프롬프트 구성
# ============================================================
def build_system_prompt(context: str = "", frames: list = None) -> str:
    """시스템 프롬프트 + 동적 컨텍스트 조합"""

    # 저장된 시스템 프롬프트 사용 (파인튜닝 v2에서 생성)
    if SYSTEM_PROMPT_FILE.exists():
        with open(SYSTEM_PROMPT_FILE, "r", encoding="utf-8") as f:
            base_prompt = f.read().strip()
    else:
        base_prompt = FALLBACK_SYSTEM_PROMPT.strip()

    parts = [base_prompt]

    # 선택된 프레임 카드 주입
    if frames:
        frame_text = "\n\n## 이 질문에 특히 관련 있는 분석 프레임:\n"
        for f in frames:
            frame_text += f"\n### {f['name']}\n"
            frame_text += f"{f['description']}\n"
            frame_text += f"추론 패턴: {f.get('reasoning_pattern', '')}\n"
        parts.append(frame_text)

    # 실시간 컨텍스트 주입 (Layer B)
    if context:
        context_text = (
            "\n\n## 참고 데이터 (사용자 제공):\n"
            f"{context}\n"
            "\n위 데이터를 분석에 활용하되, 데이터의 정확성은 사용자에게 확인을 요청하세요."
        )
        parts.append(context_text)

    return "\n".join(parts)


# ============================================================
# 분석 모드 — 자동 프레임 기반 분석
# ============================================================
def auto_analyze(client: OpenAI, model: str, topic: str, context: str = ""):
    """주제를 받아 자동으로 프레임 선택 + 분석"""
    print(f"\n{'='*60}")
    print(f"  🔍 자동 분석: {topic}")
    print(f"{'='*60}")

    # 1) 관련 프레임 자동 선택
    relevant = find_relevant_frames(topic)
    if relevant:
        print(f"\n  적용 프레임:")
        for f in relevant:
            print(f"    • {f['name']}")
    else:
        print(f"\n  ⚠️ 특별히 매칭되는 프레임이 없어 범용 분석을 수행합니다.")

    # 2) 시스템 프롬프트 구성
    sys_prompt = build_system_prompt(context=context, frames=relevant)

    # 3) 분석 요청 구성
    user_msg = (
        f"다음 주제/현상에 대해 송근용의 관점으로 분석해 주세요:\n\n"
        f"**주제**: {topic}\n"
    )
    if context:
        user_msg += f"\n**참고 데이터**:\n{context}\n"

    user_msg += (
        "\n다음 구조로 답변해 주세요:\n"
        "1. 현상 정리\n"
        "2. 적용 프레임과 분석\n"
        "3. 핵심 체크포인트\n"
        "4. 시나리오 (긍정/부정/베이스)\n"
        "5. 겸손 코멘트\n"
    )

    # 4) API 호출
    print(f"\n  분석 중...\n")
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.7,
            max_tokens=2000,
        )
        answer = response.choices[0].message.content
        print(f"{'─'*60}")
        print(answer)
        print(f"{'─'*60}")
        print(f"\n  모델: {model}")
        print(f"  토큰: {response.usage.total_tokens}")
    except Exception as e:
        print(f"  ❌ 오류: {e}")


# ============================================================
# 대화형 모드
# ============================================================
def interactive_mode(client: OpenAI, model: str, context: str = ""):
    """대화형 분석 세션"""
    print(f"\n{'='*60}")
    print(f"  🤖 송근용 에이전트 — 대화형 분석")
    print(f"{'='*60}")
    print(f"  모델: {model}")
    print(f"  종료: quit | 프레임 보기: /frames | 컨텍스트 추가: /ctx ...")
    if context:
        print(f"  컨텍스트: {context[:80]}...")
    print()

    history = []
    session_context = context

    while True:
        try:
            user_input = input("  질문> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue

        # 특수 명령어
        if user_input.lower() in ("quit", "exit", "q"):
            break

        if user_input == "/frames":
            show_frames()
            continue

        if user_input.startswith("/ctx "):
            new_ctx = user_input[5:].strip()
            session_context = (session_context + "\n" + new_ctx).strip() if session_context else new_ctx
            print(f"  ✅ 컨텍스트 추가됨 (총 {len(session_context)}자)")
            continue

        if user_input == "/clear":
            history = []
            print("  ✅ 대화 기록 초기화")
            continue

        if user_input == "/help":
            print("  명령어:")
            print("    /frames  — 프레임 카드 목록")
            print("    /ctx ... — 실시간 데이터 추가")
            print("    /clear   — 대화 기록 초기화")
            print("    /help    — 도움말")
            print("    quit     — 종료")
            continue

        # 관련 프레임 자동 선택
        relevant = find_relevant_frames(user_input)

        # 시스템 프롬프트 구성
        sys_prompt = build_system_prompt(
            context=session_context, frames=relevant
        )

        # 메시지 구성 (최근 10턴 유지)
        messages = [{"role": "system", "content": sys_prompt}]
        messages.extend(history[-20:])  # 최근 10턴 (20메시지)
        messages.append({"role": "user", "content": user_input})

        # API 호출
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                max_tokens=2000,
            )
            answer = response.choices[0].message.content

            # 기록 저장
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": answer})

            print(f"\n{'─'*55}")
            print(answer)
            print(f"{'─'*55}")

            # 적용된 프레임 표시
            if relevant:
                frame_names = ", ".join(f["name"] for f in relevant)
                print(f"  [프레임: {frame_names}]")
            print(f"  [토큰: {response.usage.total_tokens}]\n")

        except Exception as e:
            print(f"\n  ❌ 오류: {e}\n")

    print("\n  세션 종료.")


# ============================================================
# 모델 이름 자동 감지
# ============================================================
def detect_model() -> str:
    """finetune_state.json에서 완성된 모델 이름 감지"""
    if STATE_FILE.exists():
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            state = json.load(f)
        model = state.get("fine_tuned_model")
        if model:
            return model
    return ""


# ============================================================
# CLI
# ============================================================
def main():
    ap = argparse.ArgumentParser(
        description="송근용 에이전트 — 하이브리드 분석 엔진",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python song_agent.py                                   # 대화형 모드
  python song_agent.py --question "반도체 사이클 분석"     # 단일 질문
  python song_agent.py --mode analyze --topic "AI 반도체"  # 자동 분석
  python song_agent.py --mode frame                       # 프레임 카드 보기
  python song_agent.py --context "삼성전자 4Q OP 6.5조"   # 데이터 주입

프레임 기반 분석 흐름:
  1. 사용자 질문/주제 입력
  2. 관련 사고 프레임 카드 자동 선택
  3. 시스템 프롬프트 + 프레임 + 실시간 데이터 조합
  4. 파인튜닝 모델(또는 프로토타입)이 분석 생성
        """,
    )
    ap.add_argument("--mode", choices=["chat", "analyze", "frame"],
                    default="chat", help="실행 모드 (기본: chat)")
    ap.add_argument("--model", default="",
                    help="모델 이름 (빈칸이면 자동 감지 → 없으면 gpt-4o-mini)")
    ap.add_argument("--question", "-q", default="",
                    help="단일 질문 (대화형 대신)")
    ap.add_argument("--topic", default="",
                    help="분석할 주제 (--mode analyze 시)")
    ap.add_argument("--context", "-c", default="",
                    help="실시간 데이터/컨텍스트 (쉼표 또는 줄바꿈 구분)")

    args = ap.parse_args()

    # 프레임 보기는 API 키 불필요
    if args.mode == "frame":
        show_frames()
        return

    # API 키 확인
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("❌ OPENAI_API_KEY 환경변수가 설정되지 않았습니다!")
        print()
        print("   설정 방법 (Windows PowerShell):")
        print('   $env:OPENAI_API_KEY = "sk-여기에_API키_입력"')
        print()
        print("   설정 방법 (Windows CMD):")
        print('   set OPENAI_API_KEY=sk-여기에_API키_입력')
        print()
        print("   API 키 발급: https://platform.openai.com/api-keys")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    # 모델 결정
    model = args.model
    if not model:
        model = detect_model()
        if model:
            print(f"  🎯 파인튜닝 모델 감지: {model}")
        else:
            model = "gpt-4o-mini"
            print(f"  ℹ️ 파인튜닝 모델 없음 — 프로토타입 모드: {model}")
            print(f"     (파인튜닝 완료 후 자동으로 전환됩니다)")

    print()
    print("🤖 송근용 에이전트 v2")
    print(f"   모델: {model}")
    print()

    # 실행
    if args.mode == "analyze":
        topic = args.topic or args.question
        if not topic:
            print("  ❌ --topic 또는 --question을 지정하세요.")
            sys.exit(1)
        auto_analyze(client, model, topic, args.context)

    elif args.question:
        # 단일 질문 모드
        auto_analyze(client, model, args.question, args.context)

    else:
        # 대화형 모드
        interactive_mode(client, model, args.context)


if __name__ == "__main__":
    main()