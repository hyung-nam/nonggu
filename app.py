import os
import re
from pathlib import Path
from typing import Optional, Tuple, List

import streamlit as st
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from openai import OpenAI

# ========= 설정 =========
BASE = Path(__file__).resolve().parent
DB_DIR = BASE / "chroma_db_openai"
COLLECTION_NAME = "tosoha1_chunks_openai"

TEXT_MODEL = "gpt-5.2-pro"   # 텍스트 분석(최고급)
VISION_MODEL = "gpt-4o"      # 이미지 분석(안정)

EMBED_MODEL = "text-embedding-3-small"  # DB 인덱싱 때 사용한 값과 동일해야 함

# RAG (유사 프레임 힌트용)
N_RESULTS = 12
MAX_CTX_CHUNKS = 4
MAX_CHARS_PER_CHUNK = 650
MAX_REF_DOCS = 2
MAX_OUTPUT_TOKENS = 950

SYSTEM_PROMPT = """당신은 ‘송근용(슬기자산운용 CIO) 블로그’에서 관찰되는 사고방식/논지 전개 방식으로 “현재 시장 현상”을 해석한다.

목표:
- 과거 글을 찾아 요약하는 사서가 아니라,
- 동일한 프레임(수요/공급/사이클/CAPEX/컨콜/심리 vs 펀더멘털/병목/경쟁/마진 구조)로 현재를 분석한다.
- 업로드된 블로그 코퍼스는 ‘근거 인용’이 아니라 ‘유사 프레임 참고’로 0~2개만 사용한다.

원칙:
1) 사실(숫자/뉴스/발언)은 사용자 입력/첨부/링크 요약/이미지 관찰에서만 인용. 모르면 “자료 부족”.
2) 해석은 가능하되 단정 금지. 유보를 자연스럽게(“답은 모르지만/다만/가능성/확인 필요”).
3) 투자 지시 금지(매수/매도/목표가 금지). 체크포인트/반증조건까지만.

출력(고정):
- 첫 줄: 한 문장 결론
- 본문: 8~14줄(담백, 유보 포함)
- 체크포인트/반증조건: 3~6개 불릿(앞단 지표)
- 마지막: “유사 프레임 참고:” 링크 0~2개(없으면 ‘없음’)

금지:
- 보고서/목차 스타일(##, (1)(2)(3), ‘핵심 요약/논리 정리/근거’) 금지
- 발췌 나열/장문 인용 금지
"""

BANNED = ["##", "(1)", "(2)", "(3)", "핵심 요약", "논리 정리", "근거(글 제목/날짜/URL)", "글의 논리 정리"]

TEMPLATE_1 = """[현상 요약]
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

TEMPLATE_2 = """[현상 요약]
- (무슨 일이 벌어졌는지 2~3줄)

[팩트/숫자]
- (컨콜/가이던스/지표/가격/재고/리드타임 등 3~8개)

[시장 반응]
- (섹터/포지셔닝/심리 1~2줄)

[내 질문]
- 이걸 어떻게 프레임으로 해석해야 하고, 다음 4주에 뭘 보면 돼?
"""

def bad_format(s: str) -> bool:
    return any(x in (s or "") for x in BANNED)

# ========= URL fetch =========
def fetch_url_text(url: str, timeout: int = 10) -> Tuple[str, str]:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.7",
    }
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")
    title = (soup.title.text.strip() if soup.title and soup.title.text else url)

    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
        tag.decompose()

    text = soup.get_text("\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    snippet = text[:7000]
    return title, snippet

# ========= File parsing =========
def read_text_file_bytes(name: str, b: bytes) -> str:
    # try utf-8 then cp949
    for enc in ["utf-8", "utf-8-sig", "cp949"]:
        try:
            return b.decode(enc)
        except Exception:
            pass
    return ""

def extract_pdf_text(pdf_bytes: bytes, max_chars: int = 12000) -> str:
    from io import BytesIO
    reader = PdfReader(BytesIO(pdf_bytes))
    texts = []
    for i, page in enumerate(reader.pages[:10]):  # 10페이지까지만
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        if t.strip():
            texts.append(t.strip())
        if sum(len(x) for x in texts) > max_chars:
            break
    joined = "\n\n".join(texts)
    return joined[:max_chars]

def parse_uploaded_file(file) -> Tuple[str, str]:
    """returns (label, extracted_text_snippet)"""
    name = file.name
    b = file.read()
    ext = name.lower().split(".")[-1]

    if ext in ["png", "jpg", "jpeg", "webp"]:
        # 이미지 텍스트는 LLM이 보게 하고, 여기선 빈 문자열
        return f"{name} (image)", ""

    if ext in ["txt", "md", "csv", "json"]:
        t = read_text_file_bytes(name, b)
        t = t.strip()
        return f"{name}", t[:12000]

    if ext in ["pdf"]:
        t = extract_pdf_text(b)
        return f"{name}", t.strip()

    # 그 외는 텍스트 추출 안 함
    return f"{name}", ""

# ========= Engine load =========
@st.cache_resource
def load_engine():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY가 없습니다. PowerShell:  $env:OPENAI_API_KEY='sk-...'\n")

    vdb = chromadb.PersistentClient(path=str(DB_DIR))
    embed_fn = OpenAIEmbeddingFunction(api_key=api_key, model_name=EMBED_MODEL)
    col = vdb.get_collection(name=COLLECTION_NAME, embedding_function=embed_fn)
    llm = OpenAI(api_key=api_key)
    return col, llm

def build_context_and_refs(hits):
    ctx_blocks = []
    ref_links = []
    seen = set()

    for doc, m in hits:
        title = (m.get("title") or "").strip()
        date = (m.get("date") or "").strip()
        url = (m.get("url") or "").strip()
        heading = (m.get("heading_path") or "").strip()

        if url and url not in seen and len(ref_links) < MAX_REF_DOCS:
            seen.add(url)
            ref_links.append(f"- {title} ({date}) {url}")

        if len(ctx_blocks) < MAX_CTX_CHUNKS:
            snippet = (doc or "").strip()
            if len(snippet) > MAX_CHARS_PER_CHUNK:
                snippet = snippet[:MAX_CHARS_PER_CHUNK] + "…"
            ctx_blocks.append(f"[{title}] {date} {url}\nheading: {heading}\n{snippet}")

        if len(ref_links) >= MAX_REF_DOCS and len(ctx_blocks) >= MAX_CTX_CHUNKS:
            break

    return "\n\n---\n\n".join(ctx_blocks), "\n".join(ref_links)

def ask_frame_engine(llm: OpenAI, col, user_text: str, image_bytes: Optional[bytes],
                     url_title: str, url_text: str, file_summaries: List[Tuple[str, str]]):

    # RAG: 유사 프레임 힌트
    res = col.query(query_texts=[user_text], n_results=N_RESULTS)
    hits = []
    for i in range(len(res["ids"][0])):
        hits.append((res["documents"][0][i], res["metadatas"][0][i]))
    ctx, refs = build_context_and_refs(hits)

    extra_parts = []

    if url_text:
        extra_parts.append(f"[링크 참고]\n- 제목: {url_title}\n- URL 내용(일부):\n{url_text}")

    if file_summaries:
        blocks = []
        for label, text in file_summaries:
            if text.strip():
                blocks.append(f"- {label}\n{text}")
            else:
                blocks.append(f"- {label}\n(텍스트 추출 불가/생략)")
        extra_parts.append("[첨부파일 참고]\n" + "\n\n".join(blocks))

    if image_bytes:
        extra_parts.append("[이미지 참고]\n- 첨부 이미지를 관찰해서 ‘보이는 사실’만 요약해 활용해라.")

    extra = "\n\n".join(extra_parts)

    user_prompt = f"""목표: ‘송근용 프레임’으로 현재 현상을 해석한다. 사서처럼 발췌를 요약하지 마라.
원칙:
- 사실(숫자/뉴스/발언)은 사용자 입력/링크/첨부/이미지에서만 인용.
- 블로그 발췌는 ‘유사 프레임 힌트’로만 사용.
- 확률적/유보 톤. 단정 금지.
- 투자 지시 금지.
- 보고서/목차 스타일 금지(##,(1)(2)(3),핵심요약/논리정리/근거).

[사용자 입력]
{user_text}

{extra}

[유사 프레임 힌트(블로그 발췌)]
{ctx}

출력(고정):
- 첫 줄 한 문장 결론
- 본문 8~14줄(담백/유보)
- 체크포인트/반증조건 3~6개(불릿)
- 마지막: 유사 프레임 참고(링크 0~2개, 없으면 ‘없음’)
"""

    model = VISION_MODEL if image_bytes else TEXT_MODEL

    if image_bytes:
        resp = llm.responses.create(
            model=model,
            input=[
                {"role":"system","content": SYSTEM_PROMPT},
                {"role":"user","content":[
                    {"type":"input_text","text": user_prompt},
                    {"type":"input_image","image_bytes": image_bytes},
                ]},
            ],
            max_output_tokens=MAX_OUTPUT_TOKENS,
        )
    else:
        resp = llm.responses.create(
            model=model,
            input=[
                {"role":"system","content": SYSTEM_PROMPT},
                {"role":"user","content": user_prompt},
            ],
            max_output_tokens=MAX_OUTPUT_TOKENS,
        )

    answer = (resp.output_text or "").strip()

    if bad_format(answer):
        repair = ("다시 답해라. 보고서/목차 스타일(##,(1)(2)(3),핵심요약/논리정리/근거) 금지. "
                  "결론 1문장 → 해석 8~14줄 → 체크포인트 3~6개 → 유사 프레임 참고(0~2개).")
        if image_bytes:
            resp2 = llm.responses.create(
                model=model,
                input=[
                    {"role":"system","content": SYSTEM_PROMPT},
                    {"role":"user","content":[
                        {"type":"input_text","text": repair + "\n\n" + user_prompt},
                        {"type":"input_image","image_bytes": image_bytes},
                    ]},
                ],
                max_output_tokens=MAX_OUTPUT_TOKENS,
            )
        else:
            resp2 = llm.responses.create(
                model=model,
                input=[
                    {"role":"system","content": SYSTEM_PROMPT},
                    {"role":"user","content": repair + "\n\n" + user_prompt},
                ],
                max_output_tokens=MAX_OUTPUT_TOKENS,
            )
        answer = (resp2.output_text or answer).strip()

    if "유사 프레임 참고" not in answer:
        answer += "\n\n유사 프레임 참고:\n" + (refs if refs.strip() else "없음")

    return answer, refs, model

# ========= UI =========
st.set_page_config(page_title="송근용 프레임 채팅", layout="centered")
st.markdown(
    """
<style>
.block-container {max-width: 860px; padding-top: 1.0rem; padding-bottom: 2rem;}
/* 입력 영역을 아래쪽에 붙여보이게 */
footer {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True
)

st.title("송근용 프레임 채팅")
st.caption("OpenAI 느낌의 간결한 채팅 UI + URL/파일/이미지 첨부(드래그&드롭) + 프레임 해석")

# session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "composer" not in st.session_state:
    st.session_state.composer = ""
if "pending_url" not in st.session_state:
    st.session_state.pending_url = ""
if "url_cache" not in st.session_state:
    st.session_state.url_cache = {"url":"", "title":"", "text":""}
if "pending_image" not in st.session_state:
    st.session_state.pending_image = None
if "pending_files" not in st.session_state:
    st.session_state.pending_files = []  # list of (label, text)

# render chat
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        meta = m.get("meta") or {}
        if meta.get("model"):
            st.caption(f"model: {meta['model']}")
        if meta.get("attachments"):
            st.caption("attachments: " + ", ".join(meta["attachments"]))

# composer area (like ChatGPT input)
st.divider()

# 템플릿/첨부를 같은 “드롭다운”에 몰아넣기
with st.expander("템플릿/첨부 (클릭해서 열기) — URL / 이미지 / 파일 드래그&드롭", expanded=False):
    colA, colB = st.columns([1,1])

    with colA:
        st.subheader("템플릿")
        if st.button("템플릿 넣기(기본)", use_container_width=True):
            st.session_state.composer = TEMPLATE_1
        if st.button("템플릿 넣기(간단)", use_container_width=True):
            st.session_state.composer = TEMPLATE_2
        st.caption("템플릿을 넣은 뒤, 필요한 부분만 채워서 보내면 품질이 더 좋아짐.")

    with colB:
        st.subheader("URL")
        url = st.text_input("URL 붙여넣기", value=st.session_state.pending_url, placeholder="https://...", key="url_input")
        c1, c2 = st.columns([1,1])
        with c1:
            if st.button("URL 가져오기", use_container_width=True):
                if url.strip():
                    try:
                        title, text = fetch_url_text(url.strip())
                        st.session_state.url_cache = {"url": url.strip(), "title": title, "text": text}
                        st.success(f"가져옴: {title}")
                    except Exception as e:
                        st.error(f"URL 가져오기 실패: {e}")
        with c2:
            if st.button("URL 비우기", use_container_width=True):
                st.session_state.url_cache = {"url":"", "title":"", "text":""}
                st.success("URL 초기화")

    st.subheader("첨부 업로드 (드래그&드롭)")
    img = st.file_uploader("이미지 업로드 (png/jpg/webp)", type=["png","jpg","jpeg","webp"], key="img_upl")
    if img is not None:
        b = img.read()
        st.session_state.pending_image = b
        st.image(b, caption="업로드 이미지", use_container_width=True)

    files = st.file_uploader("파일 업로드 (txt/md/csv/json/pdf)", type=["txt","md","csv","json","pdf"], accept_multiple_files=True, key="files_upl")
    if files:
        parsed = []
        for f in files[:3]:  # 너무 많으면 컨텍스트 과부하 → 3개로 제한
            label, text = parse_uploaded_file(f)
            parsed.append((label, text))
        st.session_state.pending_files = parsed
        st.success(f"파일 {len(parsed)}개 준비됨(최대 3개 반영)")

    if st.button("첨부 초기화", use_container_width=True):
        st.session_state.pending_image = None
        st.session_state.pending_files = []
        st.session_state.url_cache = {"url":"", "title":"", "text":""}
        st.success("첨부 초기화 완료")

# message composer
st.session_state.composer = st.text_area(
    "메시지 작성",
    value=st.session_state.composer,
    height=150,
    placeholder="여기에 질문을 입력하세요. (템플릿을 쓰면 더 좋음)",
    label_visibility="collapsed",
)

send = st.button("보내기", type="primary", use_container_width=True)

if send:
    user_text = st.session_state.composer.strip()
    if not user_text:
        st.warning("메시지를 입력해줘.")
        st.stop()

    # attachments summary
    attachments = []
    url_title = st.session_state.url_cache.get("title","")
    url_text  = st.session_state.url_cache.get("text","")
    url_used  = st.session_state.url_cache.get("url","")
    if url_used:
        attachments.append("url")
    if st.session_state.pending_image is not None:
        attachments.append("image")
    if st.session_state.pending_files:
        attachments.append("file")

    # show user message
    st.session_state.messages.append({"role":"user","content":user_text, "meta":{"attachments": attachments}})

    with st.chat_message("assistant"):
        try:
            col, llm = load_engine()
            with st.spinner("분석 중..."):
                answer, refs, used_model = ask_frame_engine(
                    llm, col, user_text,
                    st.session_state.pending_image,
                    url_title, url_text,
                    st.session_state.pending_files
                )
            st.markdown(answer)
            st.caption(f"model: {used_model}")
        except Exception as e:
            answer = f"에러: {e}"
            used_model = ""
            st.error(answer)

    st.session_state.messages.append({"role":"assistant","content":answer, "meta":{"model": used_model, "attachments": attachments}})

    # reset composer + attachments after send (ChatGPT처럼 한 번 보내면 첨부는 초기화)
    st.session_state.composer = ""
    st.session_state.pending_image = None
    st.session_state.pending_files = []
    st.session_state.url_cache = {"url":"", "title":"", "text":""}
    st.rerun()
