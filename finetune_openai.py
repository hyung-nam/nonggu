#!/usr/bin/env python3
"""
OpenAI Fine-tuning 실행 스크립트
================================
변환된 학습 데이터를 OpenAI에 업로드하고 파인튜닝을 실행합니다.

사전 준비:
    1. pip install openai
    2. OpenAI API 키 발급: https://platform.openai.com/api-keys
    3. 환경변수 설정: set OPENAI_API_KEY=sk-...

사용법:
    python finetune_openai.py                          # 전체 과정 (업로드 → 파인튜닝)
    python finetune_openai.py --step upload             # 파일 업로드만
    python finetune_openai.py --step train              # 파인튜닝 시작만
    python finetune_openai.py --step status             # 진행 상황 확인
    python finetune_openai.py --step test               # 완성된 모델 테스트
"""

import argparse
import json
import os
import sys
import time
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
DEFAULT_TRAIN_FILE = Path(__file__).parent / "data" / "finetune_train.jsonl"
DEFAULT_TEST_FILE = Path(__file__).parent / "data" / "finetune_test.jsonl"
STATE_FILE = Path(__file__).parent / "data" / "finetune_state.json"

# 파인튜닝 모델 (비용 효율적인 순서)
AVAILABLE_MODELS = {
    "gpt-4o-mini-2024-07-18": "GPT-4o-mini (가장 저렴, 추천)",
    "gpt-3.5-turbo-0125": "GPT-3.5-turbo (중간)",
    "gpt-4o-2024-08-06": "GPT-4o (가장 비쌈)",
}
DEFAULT_MODEL = "gpt-4o-mini-2024-07-18"


# ============================================================
# 상태 관리
# ============================================================
def load_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_state(state: dict):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


# ============================================================
# 1단계: 파일 업로드
# ============================================================
def step_upload(client: OpenAI, train_path: Path, test_path: Path) -> dict:
    print("=" * 55)
    print("  📤 1단계: 학습 데이터 업로드")
    print("=" * 55)

    state = load_state()

    # Train 파일 업로드
    if state.get("train_file_id"):
        print(f"\n  이미 업로드됨: {state['train_file_id']}")
    else:
        print(f"\n  Train 파일: {train_path}")
        if not train_path.exists():
            print(f"  ❌ 파일 없음! 먼저 convert_to_finetune.py를 실행하세요.")
            sys.exit(1)

        # 줄 수 확인
        with open(train_path, "r", encoding="utf-8") as f:
            line_count = sum(1 for _ in f)
        print(f"  학습 예시: {line_count}개")

        if line_count < 10:
            print("  ⚠️ 학습 데이터가 10개 미만입니다. 최소 10개 이상 권장.")

        print("  업로드 중...")
        with open(train_path, "rb") as f:
            response = client.files.create(file=f, purpose="fine-tune")
        state["train_file_id"] = response.id
        print(f"  ✅ Train 업로드 완료: {response.id}")

    # Test 파일 업로드 (선택)
    if test_path.exists() and not state.get("test_file_id"):
        print(f"\n  Test 파일: {test_path}")
        print("  업로드 중...")
        with open(test_path, "rb") as f:
            response = client.files.create(file=f, purpose="fine-tune")
        state["test_file_id"] = response.id
        print(f"  ✅ Test 업로드 완료: {response.id}")

    save_state(state)
    return state


# ============================================================
# 2단계: 파인튜닝 시작
# ============================================================
def step_train(client: OpenAI, model: str, n_epochs: int, suffix: str) -> dict:
    print("\n" + "=" * 55)
    print("  🏋️ 2단계: 파인튜닝 시작")
    print("=" * 55)

    state = load_state()

    if state.get("job_id"):
        print(f"\n  이미 파인튜닝 진행 중: {state['job_id']}")
        return state

    if not state.get("train_file_id"):
        print("  ❌ 먼저 upload 단계를 실행하세요.")
        sys.exit(1)

    print(f"\n  모델: {model}")
    print(f"  에포크: {n_epochs}")
    print(f"  접미사: {suffix}")

    # 파인튜닝 작업 생성
    params = {
        "training_file": state["train_file_id"],
        "model": model,
        "suffix": suffix,
        "hyperparameters": {
            "n_epochs": n_epochs,
        },
    }

    if state.get("test_file_id"):
        params["validation_file"] = state["test_file_id"]

    job = client.fine_tuning.jobs.create(**params)

    state["job_id"] = job.id
    state["model"] = model
    state["status"] = job.status
    save_state(state)

    print(f"\n  ✅ 파인튜닝 작업 생성!")
    print(f"  Job ID: {job.id}")
    print(f"  상태:   {job.status}")
    print(f"\n  💡 진행 상황 확인:")
    print(f"     python finetune_openai.py --step status")
    print(f"     또는 https://platform.openai.com/finetune 에서 확인")

    return state


# ============================================================
# 3단계: 진행 상황 확인
# ============================================================
def step_status(client: OpenAI, wait: bool = False):
    print("\n" + "=" * 55)
    print("  📊 파인튜닝 진행 상황")
    print("=" * 55)

    state = load_state()

    if not state.get("job_id"):
        print("\n  ❌ 진행 중인 파인튜닝 작업이 없습니다.")
        print("     먼저 train 단계를 실행하세요.")
        return

    job = client.fine_tuning.jobs.retrieve(state["job_id"])
    print(f"\n  Job ID:  {job.id}")
    print(f"  모델:    {job.model}")
    print(f"  상태:    {job.status}")

    if job.fine_tuned_model:
        state["fine_tuned_model"] = job.fine_tuned_model
        save_state(state)
        print(f"  결과:    {job.fine_tuned_model}")
        print(f"\n  🎉 파인튜닝 완료!")
        print(f"     테스트: python finetune_openai.py --step test")
        return

    if job.error and job.error.message:
        print(f"  오류:    {job.error.message}")
        return

    # 이벤트 출력
    print(f"\n  최근 이벤트:")
    events = client.fine_tuning.jobs.list_events(
        fine_tuning_job_id=state["job_id"], limit=10
    )
    for event in reversed(events.data):
        print(f"    [{event.created_at}] {event.message}")

    if wait and job.status in ("validating_files", "queued", "running"):
        print(f"\n  ⏳ 완료 대기 중... (Ctrl+C로 중단 가능)")
        while True:
            time.sleep(30)
            job = client.fine_tuning.jobs.retrieve(state["job_id"])
            print(f"  상태: {job.status}", end="")
            if job.fine_tuned_model:
                state["fine_tuned_model"] = job.fine_tuned_model
                save_state(state)
                print(f" → 완료! 모델: {job.fine_tuned_model}")
                return
            elif job.status in ("failed", "cancelled"):
                print(f" → {job.status}")
                if job.error:
                    print(f"  오류: {job.error.message}")
                return
            print()


# ============================================================
# 4단계: 테스트
# ============================================================
def step_test(client: OpenAI):
    print("\n" + "=" * 55)
    print("  🧪 파인튜닝 모델 테스트")
    print("=" * 55)

    state = load_state()
    model_name = state.get("fine_tuned_model")

    if not model_name:
        print("\n  ❌ 파인튜닝이 완료되지 않았습니다.")
        print("     먼저 status를 확인하세요.")
        return

    print(f"\n  모델: {model_name}")
    print(f"  종료하려면 'quit' 입력\n")

    system_msg = (
        "당신은 '송근용 에이전트'입니다. "
        "송근용의 네이버 블로그(blog.naver.com/tosoha1)에 작성된 글을 기반으로 "
        "질문에 상세하게 답변합니다. "
        "블로그 글의 내용을 정확히 전달하되, 자연스러운 한국어로 답변하세요."
    )

    while True:
        try:
            question = input("  질문> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not question or question.lower() in ("quit", "exit", "q"):
            break

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": question},
                ],
                temperature=0.7,
                max_tokens=1000,
            )
            answer = response.choices[0].message.content
            print(f"\n  답변: {answer}\n")
        except Exception as e:
            print(f"\n  ❌ 오류: {e}\n")

    print("\n  테스트 종료.")


# ============================================================
# CLI
# ============================================================
def main():
    ap = argparse.ArgumentParser(
        description="OpenAI Fine-tuning 실행 도구",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
전체 과정:
  1. python convert_to_finetune.py           # 데이터 변환
  2. python finetune_openai.py               # 업로드 + 파인튜닝
  3. python finetune_openai.py --step status  # 진행 확인
  4. python finetune_openai.py --step test    # 완성 모델 테스트

개별 단계:
  python finetune_openai.py --step upload    # 파일 업로드만
  python finetune_openai.py --step train     # 파인튜닝 시작만
  python finetune_openai.py --step status    # 상태 확인
  python finetune_openai.py --step test      # 모델 테스트
        """,
    )
    ap.add_argument("--step", choices=["all", "upload", "train", "status", "test"],
                    default="all", help="실행할 단계 (기본: all = upload + train)")
    ap.add_argument("--input", type=Path, default=DEFAULT_TRAIN_FILE,
                    help="학습 데이터 경로")
    ap.add_argument("--test-input", type=Path, default=DEFAULT_TEST_FILE,
                    help="테스트 데이터 경로")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    choices=list(AVAILABLE_MODELS.keys()),
                    help=f"기본 모델 (기본: {DEFAULT_MODEL})")
    ap.add_argument("--epochs", type=int, default=3,
                    help="학습 에포크 수 (기본: 3)")
    ap.add_argument("--suffix", default="song-agent",
                    help="모델 이름 접미사 (기본: song-agent)")
    ap.add_argument("--wait", action="store_true",
                    help="파인튜닝 완료까지 대기")

    args = ap.parse_args()

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

    print()
    print("🤖 송근용 에이전트 — OpenAI Fine-tuning")
    print(f"   모델: {AVAILABLE_MODELS.get(args.model, args.model)}")
    print()

    if args.step in ("all", "upload"):
        step_upload(client, args.input, args.test_input)

    if args.step in ("all", "train"):
        step_train(client, args.model, args.epochs, args.suffix)

    if args.step == "status":
        step_status(client, wait=args.wait)

    if args.step == "test":
        step_test(client)

    if args.step == "all":
        print("\n" + "=" * 55)
        print("  ✅ 업로드 + 파인튜닝 시작 완료!")
        print("=" * 55)
        print()
        print("  파인튜닝은 보통 30분~2시간 소요됩니다.")
        print()
        print("  진행 확인:")
        print("    python finetune_openai.py --step status")
        print("    python finetune_openai.py --step status --wait  (완료 대기)")
        print()
        print("  완료 후 테스트:")
        print("    python finetune_openai.py --step test")
        print()


if __name__ == "__main__":
    main()