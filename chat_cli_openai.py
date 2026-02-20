import os, json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from openai import OpenAI

BASE = Path(__file__).resolve().parent
DB_DIR = BASE / "chroma_db_openai"
COLLECTION_NAME = "tosoha1_chunks_openai"

MODEL = "gpt-5.2-pro"                 # Responses API
EMBED_MODEL = "text-embedding-3-small"  # DB 인덱싱 때 사용한 임베딩과 동일해야 함

# ----- Retrieval 튜닝 -----
N_RESULTS = 24
MAX_CTX_CHUNKS = 8
MAX_CHARS_PER_CHUNK = 700
MAX_REF_DOCS = 2

# ----- 품질/재작성 -----
MAX_OUTPUT_TOKENS = 1100
CRITIC_RETRIES = 2
CRITIC_PASS_SCORE = 16   # 20점 만점 합격선(상향)

REALITY_TEMPLATE = """[현상 요약]
- 
- 
- 

[팩트/숫자]
- 
- 
- 

[시장 반응]
- 

[내 질문]
- 
"""

FRAME_CARDS = """
[프레임 카드(요약)]
- 무슨 게임인가(사이클/심리/펀더멘털)부터 분류
- 시간축(오늘/1~2분기/1~2년) 먼저 고정
- 수요 있는 공급 vs 수요 없는 공급
- 재고→가격→가동률/비트→CAPEX 연결
- CAPEX는 발주/리드타임으로 확인(말 말고 일정)
- 컨콜: 단어보다 제약(가격/마진/재고/가이던스 하단/범위)
- 경쟁: 따라오기 시작하면 마진/가격이 흔들림
- 병목: 느린 고리(공정/패키징/전력/네트워크/리드타임)
- 결론은 확률적 + 트리거/반증조건
- 링크는 참고 프레임 0~2개(사서 금지)
"""

SYSTEM_PROMPT = f"""너는 ‘송근용(슬기자산운용 CIO) 블로그’의 사고방식/논지 전개로 현재 시장 현상을 해석한다.
사서(RAG 요약기) 금지. 프레임으로 현재를 분석한다.

핵심 규칙:
1) 사실(숫자/뉴스/발언)은 사용자가 준 입력에서만 인용. 없으면 '자료 부족'이라고 말한다.
2) 해석은 가능하되 단정 금지(확률적/유보).
3) 투자 지시 금지.
4) 출력 형식(고정):
   - 한 문장 결론
   - 시나리오 A/B/C (확률 합 100%) + 각 시나리오: 핵심 논리/트리거/반증조건/관찰지표
   - 공통 체크포인트 3~6개(앞단 지표)
   - 다음 4주 관찰 플랜(주차별 또는 체크리스트)
   - 유사 프레임 참고 링크 0~2개
5) 보고서/목차 스타일(##, (1)(2)(3), 핵심요약/논리정리/근거) 금지.

내재화 프레임:
{FRAME_CARDS}
"""

BANNED = ["##", "(1)", "(2)", "(3)", "핵심 요약", "논리 정리", "근거(글 제목/날짜/URL)", "글의 논리 정리"]

def bad_format(text: str) -> bool:
    return any(x in (text or "") for x in BANNED)

def is_template_like(text: str) -> bool:
    low = text.lower()
    return ("[현상" in low) and ("[팩트" in low or "팩트" in low) and ("[내 질문" in low or "질문" in low)

def build_candidates(hits: List[Tuple[str, dict]]) -> List[dict]:
    cands = []
    for doc, m in hits[:MAX_CTX_CHUNKS]:
        title = (m.get("title") or "").strip()
        date = (m.get("date") or "").strip()
        url = (m.get("url") or "").strip()
        heading = (m.get("heading_path") or "").strip()
        snippet = (doc or "").strip()
        if len(snippet) > MAX_CHARS_PER_CHUNK:
            snippet = snippet[:MAX_CHARS_PER_CHUNK] + "…"
        cands.append({"title": title, "date": date, "url": url, "heading": heading, "snippet": snippet})
    return cands

def llm_json(llm: OpenAI, prompt: str, max_tokens: int = 900) -> Dict[str, Any]:
    resp = llm.responses.create(
        model=MODEL,
        input=[
            {"role": "system", "content": "반드시 JSON만 출력. 설명/코드블록/여분 텍스트 금지."},
            {"role": "user", "content": prompt},
        ],
        max_output_tokens=max_tokens,
    )
    txt = (resp.output_text or "").strip()
    try:
        return json.loads(txt)
    except Exception:
        s = txt.find("{"); e = txt.rfind("}")
        if s != -1 and e != -1 and e > s:
            try:
                return json.loads(txt[s:e+1])
            except Exception:
                return {}
        return {}

def select_refs(llm: OpenAI, reality: str, candidates: List[dict]) -> List[dict]:
    if not candidates:
        return []
    cand_text = "\n\n".join(
        [f"[{i+1}] {c['title']} | {c['date']} | {c['url']}\nheading:{c['heading']}\n{c['snippet']}"
         for i, c in enumerate(candidates)]
    )
    prompt = f"""사용자 입력을 해석하는 데 도움이 되는 '유사 프레임' 글 0~2개만 골라라.
정답 근거 찾기/요약이 아니다. 프레임 유사도(사이클/수요공급/CAPEX/컨콜/심리/병목/경쟁/마진 구조)로 고른다.

JSON:
{{"pick":[{{"index":번호,"reason":"한 문장"}}, ... ]}}

[현실 입력]
{reality}

[후보]
{cand_text}
"""
    data = llm_json(llm, prompt, max_tokens=450)
    picks = data.get("pick", []) if isinstance(data, dict) else []
    out = []
    for p in picks[:MAX_REF_DOCS]:
        try:
            idx = int(p.get("index", 0))
        except Exception:
            continue
        if 1 <= idx <= len(candidates):
            c = candidates[idx-1].copy()
            c["reason"] = (p.get("reason") or "").strip()
            out.append(c)
    return out

def frame_plan_abc(llm: OpenAI, reality: str, ref_snips: str) -> Dict[str, Any]:
    prompt = f"""너는 송근용 프레임으로 현상을 구조화한다. 발췌 요약 금지.
반드시 아래 JSON 스키마로만 출력하라. (prob는 합 100)

스키마:
{{
 "game":"cycle|fundamental|narrative|mixed",
 "time_horizon":"days|weeks|quarters|years",
 "thesis":"한 문장 결론(확률적)",
 "scenarios":[
   {{
     "name":"A",
     "prob":0,
     "core_frames":["FC..","FC.."],
     "logic":["문장 3~6개(핵심 논리)"],
     "triggers":["확률이 A로 이동하는 조건 2~4개"],
     "falsifiers":["A가 틀리면 먼저 깨질 것 2~3개"],
     "watch":["앞단 관찰지표 3~5개"]
   }},
   {{...B...}},
   {{...C...}}
 ],
 "common_checkpoints":["공통 체크포인트 3~6개(앞단 지표)"],
 "four_week_plan":[
   "Week1: ...",
   "Week2: ...",
   "Week3: ...",
   "Week4: ..."
 ],
 "missing_info":["추가로 알면 좋은 질문 0~5개(핵심이면 채우라고 요구)"]
}}

주의:
- 사실(숫자/뉴스/발언)은 사용자 입력에서만 인용. 없으면 '자료 부족'으로 표시.
- 해석은 가능하되 단정 금지.
- 시나리오 A/B/C는 서로 '다른 게임'이 되도록(예: 수요진짜/수요착시/수급착시 등).
- common_checkpoints는 시나리오를 가르는 지표여야 한다.

[현실 입력]
{reality}

[유사 프레임 힌트(참고 발췌)]
{ref_snips}
"""
    return llm_json(llm, prompt, max_tokens=1100)

def render_voice(llm: OpenAI, reality: str, plan: Dict[str, Any], refs: List[dict]) -> str:
    refs_lines = []
    for r in refs[:MAX_REF_DOCS]:
        refs_lines.append(f"- {r.get('title','')} ({r.get('date','')}) {r.get('url','')}  # {r.get('reason','')}")
    refs_text = "\n".join(refs_lines) if refs_lines else "없음"

    prompt = f"""아래 JSON(논지 뼈대)을 ‘송근용 톤’으로 출력한다.
금지: ##, (1)(2)(3), ‘핵심 요약/논리 정리/근거’ 같은 보고서 목차.
필수 출력 구조:
- 첫 줄: 한 문장 결론
- 시나리오 A/B/C: (확률%), 각 시나리오에 '핵심 논리(3~6문장) / 트리거 / 반증조건 / 관찰지표'를 짧게
- 공통 체크포인트 3~6개(불릿)
- 다음 4주 관찰 플랜(Week1~4 또는 체크리스트)
- 마지막: '유사 프레임 참고:' 링크 0~2개(없으면 '없음')

톤:
- 담백하고 짧은 단락, 유보 자연스럽게.
- 비교 프레임 가능(지금의 X vs 과거의 Y), 단정 금지.

[현실 입력]
{reality}

[논지 JSON]
{json.dumps(plan, ensure_ascii=False)}

[유사 프레임 참고 링크]
{refs_text}
"""
    resp = llm.responses.create(
        model=MODEL,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    return (resp.output_text or "").strip()

def critic(llm: OpenAI, reality: str, answer: str) -> Dict[str, Any]:
    prompt = f"""너는 평가자다. 아래 답변이 ‘송근용 프레임 엔진 v5’ 요구를 만족하는지 채점한다.
반드시 JSON으로만 출력.

각 0~5점:
- frame_use: 프레임/시간축/제약/사이클/컨콜/병목 등 사용
- voice: 담백/유보/확률적 톤과 논지 전개
- scenario_quality: A/B/C가 진짜로 서로 다른 시나리오이고 확률 합 100인지, 트리거/반증조건이 있는지
- checkpoints: 공통 체크포인트/4주 플랜이 실제 관찰 가능하고 앞단 지표인지

fail 조건:
- 보고서/목차 스타일 포함(##,(1)(2)(3),핵심요약/논리정리/근거)
- A/B/C 또는 4주 플랜이 누락됨

출력:
{{
 "scores":{{"frame_use":0,"voice":0,"scenario_quality":0,"checkpoints":0}},
 "total":0,
 "fail":false,
 "fix":"개선 지시 3줄"
}}

[현실 입력]
{reality}

[답변]
{answer}
"""
    data = llm_json(llm, prompt, max_tokens=450)
    if not isinstance(data, dict):
        return {"scores": {}, "total": 0, "fail": True, "fix": "채점 JSON 파싱 실패. 형식을 더 엄격히 맞춰 재작성."}
    return data

def improve(llm: OpenAI, reality: str, prev: str, fix: str, refs: List[dict]) -> str:
    refs_lines = []
    for r in refs[:MAX_REF_DOCS]:
        refs_lines.append(f"- {r.get('title','')} ({r.get('date','')}) {r.get('url','')}  # {r.get('reason','')}")
    refs_text = "\n".join(refs_lines) if refs_lines else "없음"

    prompt = f"""아래 답변을 v5 요구에 맞게 다시 써라. 특히 아래 수정 지시를 반드시 반영:
{fix}

필수:
- 한 문장 결론
- 시나리오 A/B/C (확률 합 100)
- 각 시나리오: 핵심 논리 / 트리거 / 반증조건 / 관찰지표
- 공통 체크포인트 3~6개
- 다음 4주 관찰 플랜
- 유사 프레임 참고 링크 0~2개

금지: ##, (1)(2)(3), 보고서 목차형.

[현실 입력]
{reality}

[이전 답변]
{prev}

[유사 프레임 참고 링크]
{refs_text}
"""
    resp = llm.responses.create(
        model=MODEL,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    return (resp.output_text or "").strip()

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    vdb = chromadb.PersistentClient(path=str(DB_DIR))
    embed_fn = OpenAIEmbeddingFunction(api_key=api_key, model_name=EMBED_MODEL)
    col = vdb.get_collection(name=COLLECTION_NAME, embedding_function=embed_fn)

    llm = OpenAI(api_key=api_key)

    print("\n프레임 엔진 v5 — 종료: /exit")
    print("템플릿:\n" + REALITY_TEMPLATE)

    while True:
        reality = input("Q> ").strip()
        if not reality:
            continue
        if reality.lower() in ["/exit","exit","quit","q"]:
            break

        if not is_template_like(reality):
            print("\n템플릿 형태로 입력해야 품질이 유지돼. 아래 4칸만 채워줘:\n")
            print(REALITY_TEMPLATE)
            continue

        # 1) 후보 검색
        res = col.query(query_texts=[reality], n_results=N_RESULTS)
        hits = [(res["documents"][0][i], res["metadatas"][0][i]) for i in range(len(res["ids"][0]))]
        cands = build_candidates(hits)

        # 2) 유사 프레임 참고 선택
        refs = select_refs(llm, reality, cands)
        ref_snips = "\n\n---\n\n".join(
            [f"[{r['title']}] {r['date']} {r['url']}\nheading:{r['heading']}\n{r['snippet']}" for r in refs]
        ) if refs else "(없음)"

        # 3) 구조화 플랜(A/B/C)
        plan = frame_plan_abc(llm, reality, ref_snips)
        missing = plan.get("missing_info", []) if isinstance(plan, dict) else []
        if isinstance(missing, list) and len(missing) >= 4:
            qs = "\n".join([f"- {x}" for x in missing[:6]])
            print("\n자료가 부족해서 확률이 흔들릴 수 있어. 아래를 알면 더 선명해져:\n" + qs + "\n")
            continue

        # 4) 문장화
        answer = render_voice(llm, reality, plan if isinstance(plan, dict) else {}, refs)

        # 5) Critic loop
        for _ in range(CRITIC_RETRIES + 1):
            sc = critic(llm, reality, answer)
            total = int(sc.get("total", 0)) if isinstance(sc, dict) else 0
            fail = bool(sc.get("fail", True)) if isinstance(sc, dict) else True
            fix = (sc.get("fix","") if isinstance(sc, dict) else "").strip()

            if (not fail) and total >= CRITIC_PASS_SCORE and (not bad_format(answer)):
                break
            if not fix:
                fix = "A/B/C를 더 분명히 구분하고(확률 합 100), 트리거/반증조건/관찰지표를 앞단 지표로 선명히. 4주 플랜 포함."
            answer = improve(llm, reality, answer, fix, refs)

        # 링크 섹션 보정
        if "유사 프레임 참고" not in answer:
            lines = []
            for r in refs[:MAX_REF_DOCS]:
                lines.append(f"- {r.get('title','')} ({r.get('date','')}) {r.get('url','')}")
            answer += "\n\n유사 프레임 참고:\n" + ("\n".join(lines) if lines else "없음")

        print("\n" + answer + "\n")

if __name__ == "__main__":
    main()
