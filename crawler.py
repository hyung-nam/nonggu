#!/usr/bin/env python3
"""
네이버 블로그 수집 파이프라인 v3 (End-to-End)
=============================================
blog.naver.com/tosoha1 → RAG 코퍼스 생성

★ v3 핵심 변경:
  - 1단계 Indexer: PostTitleListAsync 전체 페이지네이션으로 logNo 완전 수집
    → categoryNo=0 전체 시도 → 부족하면 카테고리별 순회 + dedup
  - 2단계 Extractor: PostView.naver 로 본문 추출
  - RSS는 50개 제한이므로 보조 수단으로만 사용

수집 모드:
    --mode list-api      PostTitleListAsync 전체 페이지네이션 (기본, 추천)
    --mode rss           RSS 피드만 (최대 50개)
    --mode search-api    네이버 검색 API (Plan B)

전체 옵션:
    python crawler.py --help

요구 패키지:
    pip install requests beautifulsoup4 lxml feedparser
"""

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

# ─── feedparser는 선택적 ───
try:
    import feedparser
    HAS_FEEDPARSER = True
except ImportError:
    import xml.etree.ElementTree as ET
    HAS_FEEDPARSER = False

# ============================================================
# 설정
# ============================================================
DEFAULT_BLOG_ID = "tosoha1"

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
LOG_DIR = BASE_DIR / "logs"
CHECKPOINT_FILE = DATA_DIR / ".checkpoint.json"
INDEX_FILE = DATA_DIR / "post_index.json"
CORPUS_FILE = DATA_DIR / "corpus.jsonl"
CHUNKS_FILE = DATA_DIR / "chunks.jsonl"

DEFAULT_DELAY = 1.5
MAX_RETRIES = 5
BACKOFF_FACTOR = 2
REQUEST_TIMEOUT = 15

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://blog.naver.com/",
    "DNT": "1",
}

BLOCK_SIGNALS = [
    "captcha", "보안문자", "비정상적인 접근", "자동화된 요청",
    "접근이 차단", "잠시 후 다시", "bot detection",
]


# ============================================================
# 로깅
# ============================================================
def setup_logging(verbose: bool = False):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"crawl_{ts}.log"

    level = logging.DEBUG if verbose else logging.INFO

    root = logging.getLogger()
    root.handlers.clear()

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger("crawler")


log = logging.getLogger("crawler")


# ============================================================
# 예외
# ============================================================
class BlockDetectedError(Exception):
    """403/캡차 등 차단 감지"""
    pass


# ============================================================
# 유틸리티
# ============================================================
class RateLimiter:
    def __init__(self, delay: float = DEFAULT_DELAY):
        self.delay = delay
        self._last = 0.0

    def wait(self):
        elapsed = time.time() - self._last
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self._last = time.time()


class Checkpoint:
    """원자적 체크포인트 (크래시 복구)"""
    def __init__(self, path: Path):
        self.path = path
        self.data = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                log.warning("체크포인트 손상 — 새로 시작")
        return {"collected_urls": [], "total_posts": 0}

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
        tmp.replace(self.path)

    @property
    def collected_urls(self) -> set:
        return set(self.data.get("collected_urls", []))

    def add_url(self, url: str):
        urls = self.data.setdefault("collected_urls", [])
        if url not in urls:
            urls.append(url)
            self.data["total_posts"] = len(urls)


# ============================================================
# 네트워크
# ============================================================
def _check_for_block(resp: requests.Response):
    if resp.status_code == 403:
        raise BlockDetectedError(
            f"HTTP 403: {resp.url}\n"
            "⛔ 차단 감지! 브라우저로 확인하세요."
        )
    if resp.status_code == 200:
        snippet = resp.text[:3000].lower()
        for signal in BLOCK_SIGNALS:
            if signal in snippet:
                raise BlockDetectedError(
                    f"차단 키워드 '{signal}': {resp.url}\n"
                    "⛔ 캡차/차단 의심!"
                )


def fetch_with_retry(url: str, session: requests.Session,
                     rl: RateLimiter) -> Optional[requests.Response]:
    rl.wait()

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(url, timeout=REQUEST_TIMEOUT)
            _check_for_block(resp)

            if resp.status_code == 200:
                return resp
            elif resp.status_code == 429:
                wait = BACKOFF_FACTOR ** attempt
                log.warning(f"429 → {wait}초 대기 ({attempt}/{MAX_RETRIES})")
                time.sleep(wait)
            elif resp.status_code == 404:
                log.warning(f"404: {url}")
                return None
            else:
                log.warning(f"HTTP {resp.status_code}: {url} ({attempt}/{MAX_RETRIES})")
                time.sleep(BACKOFF_FACTOR ** attempt)

        except BlockDetectedError:
            raise
        except requests.exceptions.Timeout:
            log.warning(f"Timeout: {url} ({attempt}/{MAX_RETRIES})")
            time.sleep(BACKOFF_FACTOR ** attempt)
        except requests.exceptions.ConnectionError as e:
            log.error(f"연결 오류: {url} → {e}")
            time.sleep(BACKOFF_FACTOR ** attempt)
        except Exception as e:
            log.error(f"예외: {url} → {e}")
            return None

    log.error(f"최대 재시도 초과: {url}")
    return None


def sanitize_filename(text: str, max_len: int = 80) -> str:
    text = re.sub(r'[\\/*?:"<>|\r\n]', "", text)
    text = re.sub(r"\s+", "_", text.strip())
    return text[:max_len] if text else "untitled"


def generate_doc_id(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()[:12]


# ============================================================
# robots.txt 가드
# ============================================================
class RobotsGuard:
    def __init__(self, session: requests.Session, rl: RateLimiter):
        self.session = session
        self.rl = rl
        self._parsers: dict = {}

    def check_domain(self, base_url: str) -> bool:
        robots_url = f"{base_url.rstrip('/')}/robots.txt"
        log.info(f"  robots.txt 확인: {robots_url}")
        self.rl.wait()

        try:
            resp = self.session.get(robots_url, timeout=10)
        except Exception as e:
            log.error(f"  robots.txt 접근 실패: {e} → 보수적 중단")
            return False

        if resp.status_code == 404:
            log.info("  robots.txt 없음 (404) → 제한 없음")
            return True
        elif resp.status_code != 200:
            log.error(f"  robots.txt HTTP {resp.status_code} → 보수적 중단")
            return False

        parser = RobotFileParser()
        parser.parse(resp.text.strip().split("\n"))
        self._parsers[base_url] = parser
        log.info(f"  robots.txt 파싱 완료: {base_url}")
        return True

    def is_allowed(self, url: str) -> bool:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        parser = self._parsers.get(base)
        if parser is None:
            return True
        ua = HEADERS["User-Agent"]
        return parser.can_fetch(ua, url) and parser.can_fetch("*", url)


# ============================================================
# ★ 1단계: 인덱서 — PostTitleListAsync 완전 수집
# ============================================================
class PostIndexer:
    """
    PostTitleListAsync.naver를 페이지네이션하여 전체 logNo를 수집.
    전략:
      1) categoryNo=0 (전체) 으로 먼저 전체 페이지 순회
      2) 수집 결과가 기대보다 적으면 → 카테고리 목록을 받아서
         카테고리별로 각각 끝까지 페이지네이션 → logNo dedup
    """

    def __init__(self, blog_id: str, session, rl, robots: RobotsGuard):
        self.blog_id = blog_id
        self.session = session
        self.rl = rl
        self.robots = robots
        self.categories: dict = {}   # catNo → catName

    def run(self, expected_total: int = 0) -> list:
        """
        전체 인덱스 수집.
        expected_total: 기대 글 수 (0이면 검증 스킵).
        반환: [{"log_no": str, "title": str, "url": str, "category": str}, ...]
        """
        log.info("=" * 55)
        log.info("  ★ 1단계: 글 인덱스 수집 (PostTitleListAsync)")
        log.info("=" * 55)

        # 기존 인덱스 파일이 있으면 로드
        if INDEX_FILE.exists():
            try:
                with open(INDEX_FILE, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                if isinstance(saved, list) and len(saved) > 0:
                    log.info(f"  기존 인덱스 파일 발견: {len(saved)}개")
                    if expected_total > 0 and len(saved) >= expected_total * 0.95:
                        log.info(f"  기대치({expected_total})의 95% 이상 → 기존 인덱스 사용")
                        return saved
                    else:
                        log.info(f"  기대치 미달 또는 미설정 → 재수집")
            except (json.JSONDecodeError, IOError):
                pass

        # 방법 1: categoryNo=0 전체
        log.info("")
        log.info("  [방법1] categoryNo=0 (전체 카테고리)로 페이지네이션")
        all_posts = self._paginate_category(cat_no=0, cat_name="전체")

        log.info(f"  → categoryNo=0 결과: {len(all_posts)}개 logNo")

        # 방법 2: 부족하면 카테고리별 순회
        if expected_total > 0 and len(all_posts) < expected_total * 0.9:
            log.info("")
            log.info(f"  [방법2] 기대치({expected_total})의 90% 미달 → 카테고리별 순회")
            self._fetch_categories()

            if self.categories:
                cat_posts = self._paginate_all_categories()
                # 기존 + 카테고리별 dedup 병합
                all_posts = self._merge_dedup(all_posts, cat_posts)
                log.info(f"  → 병합 후: {len(all_posts)}개 logNo (dedup 완료)")
        elif len(all_posts) == 0:
            # categoryNo=0이 아예 빈 경우도 카테고리별 시도
            log.info("")
            log.info("  [방법2] categoryNo=0 결과 없음 → 카테고리별 순회")
            self._fetch_categories()
            if self.categories:
                all_posts = self._paginate_all_categories()
                log.info(f"  → 카테고리별 결과: {len(all_posts)}개 logNo")

        # 인덱스 저장
        self._save_index(all_posts)

        log.info("")
        log.info(f"  ✅ 최종 인덱스: {len(all_posts)}개 글")
        if expected_total > 0:
            coverage = len(all_posts) / expected_total * 100
            log.info(f"     기대치 대비: {coverage:.1f}% ({len(all_posts)}/{expected_total})")
        log.info("")

        return all_posts

    def _paginate_category(self, cat_no: int, cat_name: str,
                           count_per_page: int = 30,
                           max_pages: int = 500) -> list:
        """한 카테고리의 전체 페이지를 순회하여 logNo 수집"""
        posts = []
        seen_lognos = set()
        consecutive_empty = 0

        for page in range(1, max_pages + 1):
            url = (
                f"https://blog.naver.com/PostTitleListAsync.naver"
                f"?blogId={self.blog_id}"
                f"&currentPage={page}"
                f"&categoryNo={cat_no}"
                f"&parentCategoryNo=0"
                f"&countPerPage={count_per_page}"
            )

            if not self.robots.is_allowed(url):
                log.warning(f"    robots.txt 차단: PostTitleListAsync")
                break

            resp = fetch_with_retry(url, self.session, self.rl)
            if not resp:
                log.warning(f"    페이지 {page} 응답 없음 → 종료")
                break

            # logNo 추출
            page_lognos = set(re.findall(r"logNo=(\d+)", resp.text))
            # blogId/숫자 패턴도 수집
            for m in re.finditer(rf"/{self.blog_id}/(\d{{10,}})", resp.text):
                page_lognos.add(m.group(1))

            # 새 logNo만 필터
            new_lognos = page_lognos - seen_lognos

            if not new_lognos:
                consecutive_empty += 1
                if consecutive_empty >= 3:
                    log.info(f"    [{cat_name}] 페이지 {page}: 3회 연속 빈 페이지 → 완료")
                    break
                continue
            else:
                consecutive_empty = 0

            seen_lognos.update(new_lognos)

            # 제목 추출 시도 (HTML 파싱)
            soup = BeautifulSoup(resp.text, "lxml")
            title_map = {}
            for a in soup.find_all("a", href=True):
                href = a["href"]
                m = re.search(r"logNo=(\d+)", href)
                if not m:
                    m = re.search(rf"/{self.blog_id}/(\d{{10,}})", href)
                if m:
                    lno = m.group(1)
                    txt = a.get_text(strip=True)
                    if txt and lno not in title_map:
                        title_map[lno] = txt

            for lno in new_lognos:
                posts.append({
                    "log_no": lno,
                    "title": title_map.get(lno, f"post_{lno}"),
                    "url": f"https://blog.naver.com/{self.blog_id}/{lno}",
                    "category": cat_name if cat_no != 0 else "",
                })

            if page % 10 == 0 or page <= 3:
                log.info(f"    [{cat_name}] 페이지 {page}: +{len(new_lognos)} (누적 {len(posts)})")

        return posts

    def _fetch_categories(self):
        """NBlogBoardCategoryListAsync로 카테고리 목록 수집"""
        url = (
            f"https://blog.naver.com/NBlogBoardCategoryListAsync.naver"
            f"?blogId={self.blog_id}"
        )
        log.info(f"  카테고리 목록 수집: {url}")

        resp = fetch_with_retry(url, self.session, self.rl)
        if not resp:
            log.warning("  카테고리 목록 수집 실패")
            return

        # categoryNo 추출
        cats = {}
        soup = BeautifulSoup(resp.text, "lxml")

        # 방법 1: <a> 태그에서 categoryNo 추출
        for a in soup.find_all("a", href=True):
            href = a["href"]
            m = re.search(r"categoryNo=(\d+)", href)
            if m:
                cat_no = m.group(1)
                cat_name = a.get_text(strip=True)
                if cat_no != "0" and cat_name:
                    # 숫자 접미사 제거 (예: "일상 (52)" → "일상")
                    cat_name_clean = re.sub(r"\s*\(\d+\)\s*$", "", cat_name).strip()
                    if cat_name_clean:
                        cats[cat_no] = cat_name_clean

        # 방법 2: 정규식 fallback
        if not cats:
            for m in re.finditer(r"categoryNo=(\d+)", resp.text):
                cat_no = m.group(1)
                if cat_no != "0" and cat_no not in cats:
                    cats[cat_no] = f"cat_{cat_no}"

        self.categories = cats
        log.info(f"  카테고리 {len(cats)}개 발견: {list(cats.values())[:10]}...")

    def _paginate_all_categories(self) -> list:
        """모든 카테고리를 각각 끝까지 페이지네이션"""
        all_posts = []
        seen_lognos = set()

        for cat_no, cat_name in self.categories.items():
            log.info(f"  카테고리 [{cat_name}] (no={cat_no}) 순회 중...")
            cat_posts = self._paginate_category(
                cat_no=int(cat_no), cat_name=cat_name
            )

            # dedup
            new_count = 0
            for p in cat_posts:
                if p["log_no"] not in seen_lognos:
                    seen_lognos.add(p["log_no"])
                    all_posts.append(p)
                    new_count += 1

            log.info(f"    → [{cat_name}]: {len(cat_posts)}개 중 {new_count}개 신규")

        return all_posts

    def _merge_dedup(self, base: list, extra: list) -> list:
        """두 리스트를 log_no 기준 dedup 병합"""
        seen = set()
        result = []
        for p in base:
            if p["log_no"] not in seen:
                seen.add(p["log_no"])
                result.append(p)
        for p in extra:
            if p["log_no"] not in seen:
                seen.add(p["log_no"])
                result.append(p)
        return result

    def _save_index(self, posts: list):
        """인덱스를 JSON 파일로 저장"""
        INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = INDEX_FILE.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(posts, f, ensure_ascii=False, indent=2)
        tmp.replace(INDEX_FILE)
        log.info(f"  인덱스 저장: {INDEX_FILE} ({len(posts)}개)")


# ============================================================
# RSS 수집기 (보조 — 최대 50개)
# ============================================================
class RSSCollector:
    def __init__(self, blog_id: str, session, rl):
        self.blog_id = blog_id
        self.rss_url = f"https://rss.blog.naver.com/{blog_id}.xml"
        self.session = session
        self.rl = rl

    def fetch_post_list(self) -> list:
        log.info(f"[RSS] 피드 요청: {self.rss_url}")
        log.info(f"[RSS] ⚠️ RSS는 최근 50개만 포함됩니다")
        resp = fetch_with_retry(self.rss_url, self.session, self.rl)
        if not resp:
            log.warning("[RSS] 접근 실패")
            return []

        posts = []

        if HAS_FEEDPARSER:
            feed = feedparser.parse(resp.text)
            for entry in feed.entries:
                url = entry.get("link", "")
                m = re.search(r"/(\d{10,})", url) or re.search(r"logNo=(\d+)", url)
                log_no = m.group(1) if m else ""
                posts.append({
                    "log_no": log_no,
                    "title": entry.get("title", ""),
                    "url": url,
                    "date": entry.get("published", ""),
                    "category": entry.get("category", ""),
                })
        else:
            try:
                root = ET.fromstring(resp.text)
                for item in root.findall(".//item"):
                    url = item.findtext("link", "")
                    m = re.search(r"/(\d{10,})", url) or re.search(r"logNo=(\d+)", url)
                    log_no = m.group(1) if m else ""
                    posts.append({
                        "log_no": log_no,
                        "title": item.findtext("title", ""),
                        "url": url,
                        "date": item.findtext("pubDate", ""),
                        "category": item.findtext("category", ""),
                    })
            except Exception as e:
                log.error(f"[RSS] 파싱 실패: {e}")
                return []

        log.info(f"[RSS] {len(posts)}개 글 수집 (RSS 최대 50개)")
        return posts


# ============================================================
# 네이버 검색 API 수집기 (Plan B)
# ============================================================
class NaverSearchAPICollector:
    API_URL = "https://openapi.naver.com/v1/search/blog.json"

    def __init__(self, blog_id, client_id, client_secret, session, rl):
        self.blog_id = blog_id
        self.client_id = client_id
        self.client_secret = client_secret
        self.session = session
        self.rl = rl

    def fetch_post_list(self, max_results=1000) -> list:
        log.info(f"[SearchAPI] 최대 {max_results}건 수집")
        posts = []
        seen = set()
        start = 1

        while start <= min(max_results, 1000):
            self.rl.wait()
            params = {
                "query": f"site:blog.naver.com/{self.blog_id}",
                "display": 100, "start": start, "sort": "date",
            }
            headers = {
                **HEADERS,
                "X-Naver-Client-Id": self.client_id,
                "X-Naver-Client-Secret": self.client_secret,
            }
            try:
                resp = self.session.get(self.API_URL, params=params,
                                        headers=headers, timeout=15)
                if resp.status_code != 200:
                    log.error(f"[SearchAPI] HTTP {resp.status_code}")
                    break
                items = resp.json().get("items", [])
                if not items:
                    break
                for item in items:
                    url = item.get("link", "")
                    if self.blog_id in url and url not in seen:
                        seen.add(url)
                        m = re.search(r"/(\d{10,})", url)
                        log_no = m.group(1) if m else ""
                        posts.append({
                            "log_no": log_no,
                            "title": re.sub(r"<.*?>", "", item.get("title", "")),
                            "url": url,
                            "date": item.get("postdate", ""),
                            "category": "",
                        })
                start += 100
            except Exception as e:
                log.error(f"[SearchAPI] {e}")
                break

        log.info(f"[SearchAPI] {len(posts)}개 URL 수집")
        return posts


# ============================================================
# ★ 2단계: 본문 추출기 (PostView.naver 기반)
# ============================================================
class PostExtractor:
    """
    허용된 경로만 사용하는 본문 추출기.
    전략 순서:
      1) mobile    — m.blog.naver.com/{blogId}/{logNo}
      2) postview  — blog.naver.com/PostView.naver?blogId=...&logNo=...
      3) og_meta   — og:description (최후 수단)
    """

    STRATEGIES = ["mobile", "postview", "og_meta"]

    def __init__(self, blog_id: str, session, rl, robots: RobotsGuard):
        self.blog_id = blog_id
        self.session = session
        self.rl = rl
        self.robots = robots

    def extract(self, post_info: dict) -> Optional[dict]:
        log_no = post_info.get("log_no", "")
        post_url = post_info.get("url", "")

        if not log_no:
            m = re.search(r"/(\d{10,})", post_url) or re.search(r"logNo=(\d+)", post_url)
            if not m:
                log.warning(f"  logNo 추출 실패: {post_url}")
                return None
            log_no = m.group(1)

        for strategy in self.STRATEGIES:
            try:
                result = getattr(self, f"_try_{strategy}")(log_no)
                if result and result.get("content_text", "").strip():
                    result["extraction_method"] = strategy
                    result["log_no"] = log_no
                    result["url"] = post_url
                    log.info(f"  ✅ [{strategy}] {len(result['content_text'])}자")
                    return result
                else:
                    log.debug(f"  [{strategy}] 본문 없음")
            except BlockDetectedError:
                raise
            except Exception as e:
                log.debug(f"  [{strategy}] 실패: {e}")

        log.warning(f"  ❌ 모든 전략 실패: {post_url}")
        return None

    def _try_mobile(self, log_no: str) -> Optional[dict]:
        url = f"https://m.blog.naver.com/{self.blog_id}/{log_no}"
        if not self.robots.is_allowed(url):
            return None
        resp = fetch_with_retry(url, self.session, self.rl)
        if not resp:
            return None
        return self._parse_html(resp.text)

    def _try_postview(self, log_no: str) -> Optional[dict]:
        url = (
            f"https://blog.naver.com/PostView.naver"
            f"?blogId={self.blog_id}&logNo={log_no}&redirect=Dlog"
        )
        if not self.robots.is_allowed(url):
            return None
        resp = fetch_with_retry(url, self.session, self.rl)
        if not resp:
            return None
        return self._parse_html(resp.text)

    def _try_og_meta(self, log_no: str) -> Optional[dict]:
        url = f"https://blog.naver.com/{self.blog_id}/{log_no}"
        resp = fetch_with_retry(url, self.session, self.rl)
        if not resp:
            return None

        soup = BeautifulSoup(resp.text, "lxml")
        og_title = soup.select_one("meta[property='og:title']")
        og_desc = soup.select_one("meta[property='og:description']")

        title = og_title.get("content", "") if og_title else ""
        desc = og_desc.get("content", "") if og_desc else ""
        if not desc:
            return None

        return {
            "title": title or f"post_{log_no}",
            "date": "", "category": "",
            "content_text": desc, "content_html": "",
        }

    def _parse_html(self, html: str) -> Optional[dict]:
        soup = BeautifulSoup(html, "lxml")

        # 제목
        title = ""
        for sel in [
            "div.se-module-text h3", "h3.se_textarea",
            "div.htitle span", "div.pcol1 span.itemSubjectBoldfont",
            "h2.tit_h2", "div.se-title-text span",
            "meta[property='og:title']", "title",
        ]:
            el = soup.select_one(sel)
            if el:
                title = el.get("content", "") if el.name == "meta" else el.get_text(strip=True)
                if title:
                    break

        # 날짜
        date_str = ""
        for sel in [
            "span.se_publishDate", "p.date", "span.date",
            "div.blog_date span", "span.se_date",
            "div._postingDate span",
            "meta[property='article:published_time']",
        ]:
            el = soup.select_one(sel)
            if el:
                date_str = el.get("content", "") if el.name == "meta" else el.get_text(strip=True)
                if date_str:
                    break

        # 카테고리
        category = ""
        for sel in ["div.blog2_category a", "a.category", "span.category",
                     "em.category", "a._categoryName"]:
            el = soup.select_one(sel)
            if el:
                category = el.get_text(strip=True)
                if category:
                    break

        # 본문
        content_html = ""
        content_text = ""

        se3 = soup.select_one("div.se-main-container")
        if se3:
            content_html = str(se3)
            content_text = self._se3_to_text(se3)
        else:
            for sel in [
                "div#postViewArea", "div.se_component_wrap",
                "div.__se_component_area", "div.post-view",
                "div#post-view", "div.post_ct",
                "div#viewTypeSelector",
                "div.sect_dsc.__viewer_container",
            ]:
                area = soup.select_one(sel)
                if area and len(area.get_text(strip=True)) > 30:
                    content_html = str(area)
                    content_text = area.get_text(separator="\n", strip=True)
                    break

        if not content_text:
            og = soup.select_one("meta[property='og:description']")
            if og:
                content_text = og.get("content", "")

        return {
            "title": title, "date": date_str, "category": category,
            "content_text": content_text, "content_html": content_html,
        }

    def _se3_to_text(self, container) -> str:
        parts = []
        for mod in container.select("div.se-module"):
            cls = mod.get("class", [])
            if "se-module-text" in cls:
                h = mod.select_one("h2, h3, h4")
                if h:
                    prefix = "#" * int(h.name[1])
                    parts.append(f"{prefix} {h.get_text(strip=True)}")
                else:
                    t = mod.get_text(separator="\n", strip=True)
                    if t:
                        parts.append(t)
            elif "se-module-image" in cls:
                img = mod.select_one("img")
                if img:
                    src = img.get("data-lazy-src") or img.get("src", "")
                    alt = img.get("alt", "이미지")
                    parts.append(f"[이미지: {alt}]({src})")
            elif "se-module-oglink" in cls:
                a = mod.select_one("a")
                if a:
                    parts.append(f"[링크: {a.get_text(strip=True)}]({a.get('href', '')})")
            elif "se-module-code" in cls:
                code = mod.get_text(separator="\n", strip=True)
                if code:
                    parts.append(f"```\n{code}\n```")
            elif "se-module-table" in cls:
                t = mod.get_text(separator=" | ", strip=True)
                if t:
                    parts.append(t)
        return "\n\n".join(parts)


# ============================================================
# Markdown 변환
# ============================================================
def to_markdown(post: dict) -> str:
    return "\n".join([
        f"# {post.get('title', 'Untitled')}",
        "",
        f"- **날짜**: {post.get('date', 'N/A')}",
        f"- **카테고리**: {post.get('category', 'N/A')}",
        f"- **원문**: {post.get('url', '')}",
        f"- **추출 방법**: {post.get('extraction_method', 'N/A')}",
        "",
        "---",
        "",
        post.get("content_text", ""),
    ])


# ============================================================
# 텍스트 청킹 (heading_path 포함)
# ============================================================
def chunk_text(text: str, title: str = "",
               chunk_size: int = 1000, overlap: int = 200) -> list:
    if not text.strip():
        return []

    current_headings = []

    raw_chunks = []
    if len(text) <= chunk_size:
        raw_chunks = [text]
    else:
        start = 0
        while start < len(text):
            end = start + chunk_size
            if end < len(text):
                boundary = -1
                for sep in ["\n\n", "\n", ". ", "! ", "? "]:
                    idx = text.rfind(sep, start + chunk_size // 2, end + 50)
                    if idx > boundary:
                        boundary = idx + len(sep)
                if boundary > start:
                    end = boundary
            chunk = text[start:end].strip()
            if chunk:
                raw_chunks.append(chunk)
            start = end - overlap

    result = []
    heading_re = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)

    for piece in raw_chunks:
        for m in heading_re.finditer(piece):
            level = len(m.group(1))
            heading = m.group(2).strip()
            current_headings = current_headings[:level - 1]
            current_headings.append(heading)

        heading_path = " > ".join([title] + current_headings) if current_headings else title
        result.append({"text": piece, "heading_path": heading_path})

    return result


# ============================================================
# 메인 파이프라인
# ============================================================
class BlogCrawler:
    def __init__(self, blog_id: str, mode: str, max_posts: int,
                 resume: bool, delay: float,
                 naver_client_id: str = "", naver_client_secret: str = "",
                 expected_total: int = 0):
        self.blog_id = blog_id
        self.mode = mode
        self.max_posts = max_posts
        self.expected_total = expected_total

        for d in [RAW_DIR, PROCESSED_DIR, LOG_DIR]:
            d.mkdir(parents=True, exist_ok=True)

        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.rl = RateLimiter(delay)

        if resume:
            self.checkpoint = Checkpoint(CHECKPOINT_FILE)
        else:
            self.checkpoint = Checkpoint(CHECKPOINT_FILE)
            self.checkpoint.data = {"collected_urls": [], "total_posts": 0}

        if resume and self.checkpoint.collected_urls:
            log.info(f"체크포인트: 이전 {len(self.checkpoint.collected_urls)}개 수집됨")

        self.robots = RobotsGuard(self.session, self.rl)
        self.naver_client_id = naver_client_id
        self.naver_client_secret = naver_client_secret

        self.stats = {
            "indexed": 0, "fetched": 0, "skipped": 0, "failed": 0,
            "doc_count": 0, "chunk_count": 0,
        }

    def run(self):
        log.info(f"{'='*55}")
        log.info(f"  네이버 블로그 크롤러 v3")
        log.info(f"  블로그: {self.blog_id}")
        log.info(f"  모드:   {self.mode}")
        log.info(f"  최대:   {self.max_posts if self.max_posts > 0 else '무제한'}개")
        log.info(f"{'='*55}")

        # robots.txt 확인
        log.info("")
        log.info("▶ 0단계: robots.txt 확인")
        for domain in ["https://blog.naver.com", "https://m.blog.naver.com"]:
            if not self.robots.check_domain(domain):
                log.error(f"⛔ {domain} robots.txt 확인 실패 → 중단")
                return

        # ── 1단계: 글 인덱스 수집 ──
        post_list = self._collect_index()
        if not post_list:
            log.error("글 인덱스를 수집할 수 없습니다. 종료.")
            return

        self.stats["indexed"] = len(post_list)

        if self.max_posts > 0:
            post_list = post_list[:self.max_posts]

        log.info(f"수집 대상: {len(post_list)}개 글")

        # ── 2단계: 본문 수집 ──
        log.info("")
        log.info("▶ 2단계: 개별 글 본문 수집")
        extractor = PostExtractor(self.blog_id, self.session, self.rl, self.robots)
        corpus_docs = []
        all_chunks = []

        try:
            for i, info in enumerate(post_list, 1):
                url = info["url"]

                if url in self.checkpoint.collected_urls:
                    log.info(f"[{i}/{len(post_list)}] 스킵 (수집됨)")
                    self.stats["skipped"] += 1
                    continue

                log.info(f"[{i}/{len(post_list)}] {info.get('title', url)}")

                post = extractor.extract(info)
                if not post:
                    self.stats["failed"] += 1
                    continue

                # 메타 병합
                if info.get("date") and not post.get("date"):
                    post["date"] = info["date"]
                if info.get("category") and not post.get("category"):
                    post["category"] = info.get("category", "")
                if info.get("title") and (not post.get("title") or post["title"].startswith("post_")):
                    post["title"] = info["title"]

                self._save_raw(post)
                self._save_markdown(post)

                doc_rec, chunk_recs = self._make_records(post)
                corpus_docs.append(doc_rec)
                all_chunks.extend(chunk_recs)

                self.checkpoint.add_url(url)
                self.checkpoint.save()
                self.stats["fetched"] += 1

        except BlockDetectedError as e:
            log.error(f"\n{'!'*55}")
            log.error(str(e))
            log.error(f"{'!'*55}")
            log.error("체크포인트 저장 후 중단. 나중에 --resume으로 이어서 수집하세요.")
            self.checkpoint.save()

        # ── 3단계: JSONL 저장 ──
        log.info("")
        log.info("▶ 3단계: JSONL 저장")
        self._save_jsonl(CORPUS_FILE, corpus_docs, "corpus")
        self._save_jsonl(CHUNKS_FILE, all_chunks, "chunks")

        self._print_summary()

    def _collect_index(self) -> list:
        """모드에 따른 인덱스 수집"""
        if self.mode == "list-api":
            return PostIndexer(
                self.blog_id, self.session, self.rl, self.robots
            ).run(expected_total=self.expected_total)

        elif self.mode == "rss":
            return RSSCollector(self.blog_id, self.session, self.rl).fetch_post_list()

        elif self.mode == "search-api":
            if not self.naver_client_id or not self.naver_client_secret:
                log.error("[SearchAPI] --naver-client-id / --naver-client-secret 필요")
                return []
            return NaverSearchAPICollector(
                self.blog_id, self.naver_client_id, self.naver_client_secret,
                self.session, self.rl,
            ).fetch_post_list()

        else:
            log.error(f"알 수 없는 모드: {self.mode}")
            return []

    def _save_raw(self, post: dict):
        log_no = post.get("log_no", generate_doc_id(post["url"]))
        name = sanitize_filename(post.get("title", "untitled"))
        if post.get("content_html"):
            (RAW_DIR / f"{log_no}_{name}.html").write_text(
                post["content_html"], encoding="utf-8")
        (RAW_DIR / f"{log_no}_{name}.txt").write_text(
            post.get("content_text", ""), encoding="utf-8")

    def _save_markdown(self, post: dict):
        log_no = post.get("log_no", generate_doc_id(post["url"]))
        name = sanitize_filename(post.get("title", "untitled"))
        (PROCESSED_DIR / f"{log_no}_{name}.md").write_text(
            to_markdown(post), encoding="utf-8")

    def _make_records(self, post: dict):
        doc_id = generate_doc_id(post["url"])
        text = post.get("content_text", "")
        title = post.get("title", "")

        doc_record = {
            "id": doc_id, "title": title,
            "date": post.get("date", ""),
            "url": post["url"],
            "category": post.get("category", ""),
            "content": text,
            "extraction_method": post.get("extraction_method", ""),
            "source": f"blog.naver.com/{self.blog_id}",
            "tags": [self.blog_id, "naver_blog"],
            "char_count": len(text),
        }
        self.stats["doc_count"] += 1

        chunks = chunk_text(text, title=title)
        chunk_records = []
        for idx, ch in enumerate(chunks):
            chunk_records.append({
                "id": f"{doc_id}_c{idx:03d}",
                "doc_id": doc_id,
                "title": title,
                "date": post.get("date", ""),
                "url": post["url"],
                "category": post.get("category", ""),
                "chunk_index": idx,
                "total_chunks": len(chunks),
                "heading_path": ch["heading_path"],
                "content": ch["text"],
                "source": f"blog.naver.com/{self.blog_id}",
                "tags": [self.blog_id, "naver_blog"],
            })
        self.stats["chunk_count"] += len(chunk_records)
        return doc_record, chunk_records

    def _save_jsonl(self, path: Path, records: list, label: str):
        if not records:
            log.info(f"  {label}: 새 레코드 없음")
            return

        existing_ids = set()
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        existing_ids.add(json.loads(line).get("id"))
                    except json.JSONDecodeError:
                        continue

        new = 0
        with open(path, "a", encoding="utf-8") as f:
            for rec in records:
                if rec["id"] not in existing_ids:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    new += 1

        log.info(f"  {label}: +{new}개 (총 {len(existing_ids) + new}개) → {path.name}")

    def _print_summary(self):
        log.info(f"\n{'='*55}")
        log.info("  ✅ 수집 완료 요약")
        log.info(f"{'='*55}")
        log.info(f"  인덱스:   {self.stats['indexed']}개 logNo 수집")
        log.info(f"  성공:     {self.stats['fetched']}개 글 추출")
        log.info(f"  스킵:     {self.stats['skipped']}개 (중복/resume)")
        log.info(f"  실패:     {self.stats['failed']}개")
        log.info(f"  문서:     {self.stats['doc_count']}개 → {CORPUS_FILE.name}")
        log.info(f"  청크:     {self.stats['chunk_count']}개 → {CHUNKS_FILE.name}")
        log.info(f"  원본:     {RAW_DIR}")
        log.info(f"  마크다운: {PROCESSED_DIR}")
        log.info(f"  인덱스:   {INDEX_FILE}")
        log.info(f"{'='*55}")


# ============================================================
# CLI
# ============================================================
def main():
    ap = argparse.ArgumentParser(
        description="네이버 블로그 → RAG 코퍼스 생성 파이프라인 v3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 1345개 전체 수집 (추천)
  python crawler.py --mode list-api --expected-total 1345

  # 테스트: 인덱스 전체 + 본문 5개만
  python crawler.py --mode list-api --max-posts 5

  # 이어서 수집 (체크포인트 resume)
  python crawler.py --mode list-api --resume

  # RSS (최대 50개)
  python crawler.py --mode rss

  # 네이버 검색 API
  python crawler.py --mode search-api \\
      --naver-client-id XXXXX \\
      --naver-client-secret YYYYY
        """,
    )
    ap.add_argument(
        "--mode", choices=["list-api", "rss", "search-api"], default="list-api",
        help="수집 모드 (기본: list-api = PostTitleListAsync 전체 페이지네이션)",
    )
    ap.add_argument("--blog-id", default=DEFAULT_BLOG_ID,
                     help=f"블로그 ID (기본: {DEFAULT_BLOG_ID})")
    ap.add_argument("--max-posts", type=int, default=0,
                     help="최대 수집 글 수 (0=무제한)")
    ap.add_argument("--resume", action="store_true",
                     help="체크포인트에서 이어서 수집")
    ap.add_argument("--delay", type=float, default=DEFAULT_DELAY,
                     help=f"요청 간 지연 초 (기본: {DEFAULT_DELAY})")
    ap.add_argument("--verbose", action="store_true",
                     help="디버그 로그")
    ap.add_argument("--expected-total", type=int, default=0,
                     help="기대 글 수 (인덱서 검증용, 예: 1345)")
    ap.add_argument("--naver-client-id", default="")
    ap.add_argument("--naver-client-secret", default="")

    args = ap.parse_args()

    global log
    log = setup_logging(args.verbose)

    crawler = BlogCrawler(
        blog_id=args.blog_id,
        mode=args.mode,
        max_posts=args.max_posts,
        resume=args.resume,
        delay=args.delay,
        naver_client_id=args.naver_client_id,
        naver_client_secret=args.naver_client_secret,
        expected_total=args.expected_total,
    )

    try:
        crawler.run()
    except KeyboardInterrupt:
        log.info("\n사용자 중단 — 체크포인트 저장")
        crawler.checkpoint.save()
        sys.exit(0)
    except BlockDetectedError as e:
        log.error(f"\n{e}")
        crawler.checkpoint.save()
        sys.exit(2)
    except Exception as e:
        log.error(f"예기치 않은 오류: {e}", exc_info=True)
        crawler.checkpoint.save()
        sys.exit(1)


if __name__ == "__main__":
    main()