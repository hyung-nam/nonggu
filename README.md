\# 네이버 블로그 RAG 코퍼스 수집기 v3



`blog.naver.com/tosoha1` 공개 글을 수집하여 "송근용 에이전트"용 RAG 코퍼스를 생성하는 파이프라인입니다.



\## 빠른 시작

```bash

\# 1. 패키지 설치

pip install -r requirements.txt



\# 2. 사전 점검 (robots.txt + RSS + 접근성)

python check\_rss.py



\# 3. 수집 실행 (추천: list-api 모드)

python crawler.py                                    # 기본: list-api 모드

python crawler.py --mode list-api --max-posts 5      # 테스트: 인덱스 전체 + 본문 5개만

python crawler.py --mode list-api --expected-total 1345  # 전체 수집 + 인덱스 검증

```



\## v3 핵심 변경점



\- \*\*2단계 파이프라인\*\*: 1단계(인덱스 수집) → 2단계(본문 추출) 분리

\- \*\*PostTitleListAsync 전체 페이지네이션\*\*: RSS 50개 제한 극복, 1345개 전체 logNo 수집

\- \*\*카테고리별 폴백\*\*: categoryNo=0으로 부족하면 카테고리별 순회 + 중복 제거

\- \*\*post\_index.json 저장\*\*: 인덱스를 파일로 저장하여 재사용 (재수집 불필요)



\## 수집 모드



| 모드 | 설명 | 언제 쓰나? |

|------|------|-----------|

| `list-api` (기본) | PostTitleListAsync 전체 페이지네이션 | 1345개 전체 수집 (추천) |

| `rss` | RSS 피드만 사용 | 최근 50개만 필요할 때 |

| `search-api` | 네이버 검색 API (Plan B) | 위 방법 모두 차단일 때 |



\### list-api 모드 동작 과정



1\. \*\*1단계 — 인덱스 수집\*\* (`PostIndexer`)

&nbsp;  - `PostTitleListAsync.naver?blogId=tosoha1\&categoryNo=0` 페이지네이션으로 전체 logNo 수집

&nbsp;  - 부족하면 `NBlogBoardCategoryListAsync.naver`로 카테고리 목록 확보 → 카테고리별 순회

&nbsp;  - 중복 logNo 자동 제거

&nbsp;  - 결과를 `data/post\_index.json`에 저장 (다음 실행 시 재사용)

&nbsp;  - `--expected-total` 옵션으로 수집 완성도 검증 (95% 이상이면 재사용)



2\. \*\*2단계 — 본문 추출\*\* (`PostExtractor`)

&nbsp;  - 각 logNo에 대해 3가지 전략 순차 시도: mobile → postview → og\_meta

&nbsp;  - raw HTML, 정제 텍스트, Markdown 저장

&nbsp;  - 체크포인트 기반 이어서 수집(resume) 지원



\### 네이버 검색 API (Plan B)

```bash

python crawler.py --mode search-api \\

&nbsp;   --naver-client-id YOUR\_CLIENT\_ID \\

&nbsp;   --naver-client-secret YOUR\_CLIENT\_SECRET

```



제한사항: 하루 25,000건, `start` 최대 1000 (즉 최대 1000개 URL).



\## 산출물 구조

```

data/

├── post\_index.json         # 1단계: 수집된 logNo 인덱스

├── raw/                    # 원문 (HTML + 텍스트)

│   ├── 12345\_제목.html

│   └── 12345\_제목.txt

├── processed/              # 정제된 Markdown

│   └── 12345\_제목.md

├── corpus.jsonl            # 문서 단위 RAG 레코드

├── chunks.jsonl            # 청크 단위 RAG 레코드 (heading\_path 포함)

└── .checkpoint.json        # 수집 진행 상태 (크래시 복구)

```



\### post\_index.json (인덱스)

```json

{

&nbsp; "blog\_id": "tosoha1",

&nbsp; "collected\_at": "2026-02-19T12:00:00",

&nbsp; "total\_posts": 1345,

&nbsp; "posts": \[

&nbsp;   {"log\_no": "223456789", "title": "글 제목", "date": "2024-01-15", "category": "카테고리명"}

&nbsp; ]

}

```



\### corpus.jsonl (문서 단위)

```json

{

&nbsp; "id": "abc123def456",

&nbsp; "title": "글 제목",

&nbsp; "date": "2024-01-15",

&nbsp; "url": "https://blog.naver.com/tosoha1/223456789",

&nbsp; "category": "카테고리명",

&nbsp; "content": "전체 본문 텍스트...",

&nbsp; "extraction\_method": "mobile",

&nbsp; "source": "blog.naver.com/tosoha1",

&nbsp; "tags": \["tosoha1", "naver\_blog"],

&nbsp; "char\_count": 3200

}

```



\### chunks.jsonl (청크 단위)

```json

{

&nbsp; "id": "abc123def456\_c000",

&nbsp; "doc\_id": "abc123def456",

&nbsp; "title": "글 제목",

&nbsp; "date": "2024-01-15",

&nbsp; "url": "https://blog.naver.com/tosoha1/223456789",

&nbsp; "category": "카테고리명",

&nbsp; "chunk\_index": 0,

&nbsp; "total\_chunks": 4,

&nbsp; "heading\_path": "글 제목 > 소제목A",

&nbsp; "content": "청크 본문...",

&nbsp; "source": "blog.naver.com/tosoha1",

&nbsp; "tags": \["tosoha1", "naver\_blog"]

}

```



\## 개별 글 추출 전략 (모바일 우선)



각 글에 대해 아래 순서로 본문 추출을 시도합니다.



1\. \*\*mobile\*\* — `m.blog.naver.com` (iframe 없이 직접 콘텐츠, 파싱이 가장 안정적)

2\. \*\*postview\*\* — `PostView.naver` (PC iframe 내부 직접 접근, robots.txt 허용 경로)

3\. \*\*og\_meta\*\* — `og:description` 메타 태그 (최소 정보, 최후 수단)



\## 전체 CLI 옵션

```

--mode {list-api,rss,search-api}  수집 모드 (기본: list-api)

--blog-id ID                      블로그 ID (기본: tosoha1)

--max-posts N                     최대 수집 글 수 (0=무제한)

--resume                          체크포인트에서 이어서 수집

--delay 1.5                       요청 간 지연 (초, 기본: 1.5)

--verbose                         디버그 로그 출력

--expected-total N                기대 글 수 (인덱서 검증용, 예: 1345)

--naver-client-id ID              네이버 Open API Client ID

--naver-client-secret SECRET      네이버 Open API Client Secret

```



\## 사용 시나리오



\### 시나리오 1: 처음부터 전체 수집

```bash

python crawler.py --mode list-api --expected-total 1345

```



\### 시나리오 2: 테스트 (본문 5개만)

```bash

python crawler.py --mode list-api --max-posts 5 --verbose

```



\### 시나리오 3: 중단 후 이어서 수집

```bash

python crawler.py --mode list-api --resume

```



\### 시나리오 4: 최근 글만 빠르게

```bash

python crawler.py --mode rss

```



\## 안전장치



\- \*\*robots.txt 준수\*\*: 허용 경로만 사용 (`/PostTitleListAsync.naver`, `/NBlogBoardCategoryListAsync.naver`, `/PostView.naver`). `/PostList.naver` 등 금지 경로는 절대 접근 안 함.

\- \*\*레이트리밋\*\*: 기본 1.5초 간격, 429 → 지수 백오프 (2→4→8→16→32초)

\- \*\*403/캡차 즉시 중단\*\*: 차단 감지 시 즉시 중단

\- \*\*중복 제거\*\*: logNo 기반 + 체크포인트 기반

\- \*\*크래시 복구\*\*: 원자적 체크포인트 저장, `--resume`으로 재개

\- \*\*인덱스 재사용\*\*: `post\_index.json` → 95% 이상이면 재수집 생략



\## 법/약관/저작권 준수



\- \*\*용도\*\*: 개인 연구용/비공개 RAG 에이전트 전용. 재배포 금지.

\- \*\*robots.txt\*\*: 자동 확인, 차단 경로 미접근, 확인 불가 시 수집 중단

\- \*\*속도 제한\*\*: 1.5초 지연 + 지수 백오프 + 403/캡차 즉시 중단

\- \*\*저작권\*\*: 원저작자에게 귀속. 공개 재배포/상업 이용 불가.

\- \*\*개인정보\*\*: 공개 글만 수집. 비공개/유료 콘텐츠 미접근.



\## 환경



\- Python 3.10+

\- Windows / macOS / Linux

\- 필수: `requests`, `beautifulsoup4`, `lxml`

\- 선택: `feedparser` (RSS 파싱 품질 향상)

