#!/usr/bin/env python3
"""
네이버 블로그 사전 점검 스크립트
================================
robots.txt 파싱 → 허용/차단 판단 → RSS 확인 → 접근성 확인

실행: python check_rss.py
      python check_rss.py --blog-id OTHER_ID
"""

import re
import sys
import time
import json
import argparse
import xml.etree.ElementTree as ET
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser
from io import StringIO

import requests

# ============================================================
# 설정
# ============================================================
DEFAULT_BLOG_ID = "tosoha1"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# 우리가 접근할 경로 목록 (robots.txt로 검증할 대상)
PATHS_TO_CHECK = [
    "/PostView.naver",
    "/PostList.naver",
    "/PostTitleListAsync.naver",
    "/NBlogBoardCategoryListAsync.naver",
]


def section(title: str):
    print(f"\n{'='*64}")
    print(f"  {title}")
    print(f"{'='*64}")


# ============================================================
# 1. robots.txt 파싱 + 판단
# ============================================================
class RobotsChecker:
    """robots.txt를 가져와서 특정 경로의 허용/차단 여부를 판단"""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.robots_url = f"{self.base_url}/robots.txt"
        self.raw_text = ""
        self.status_code = None
        self.rules_summary = []   # (user-agent, [(allow/disallow, path)])
        self.parser = RobotFileParser()
        self.accessible = False

    def fetch_and_parse(self) -> bool:
        """
        robots.txt를 가져와 파싱.
        반환: True=접근 가능, False=보수적 중단 필요
        """
        section(f"1. robots.txt 확인: {self.robots_url}")

        try:
            resp = requests.get(
                self.robots_url,
                headers=HEADERS,
                timeout=10,
                allow_redirects=True,
            )
            self.status_code = resp.status_code
            print(f"  URL:    {self.robots_url}")
            print(f"  Status: {self.status_code}")

        except requests.exceptions.ConnectionError as e:
            print(f"  ❌ 연결 실패: {e}")
            print(f"  → 보수적 판단: robots.txt 확인 불가 → 수집 중단 권장")
            return False
        except requests.exceptions.Timeout:
            print(f"  ❌ 타임아웃")
            print(f"  → 보수적 판단: robots.txt 확인 불가 → 수집 중단 권장")
            return False
        except Exception as e:
            print(f"  ❌ 예외: {e}")
            print(f"  → 보수적 판단: 수집 중단 권장")
            return False

        # --- HTTP 상태 코드별 처리 ---
        if self.status_code == 200:
            self.raw_text = resp.text
            return self._parse_rules()
        elif self.status_code == 404:
            # robots.txt 없음 = 제한 없음 (RFC 9309)
            print(f"  → robots.txt 없음 (404) = 크롤링 제한 없음")
            self.accessible = True
            return True
        elif self.status_code in (403, 401):
            print(f"  → 접근 거부 ({self.status_code})")
            print(f"  → 보수적 판단: 전체 차단으로 간주 → 수집 중단 권장")
            return False
        elif self.status_code >= 500:
            print(f"  → 서버 오류 ({self.status_code})")
            print(f"  → 보수적 판단: robots.txt 확인 불가 → 수집 중단 권장")
            return False
        else:
            print(f"  → 예상 외 응답 ({self.status_code})")
            print(f"  → 보수적 판단: 수집 중단 권장")
            return False

    def _parse_rules(self) -> bool:
        """robots.txt 텍스트를 파싱하여 규칙 요약 + 경로별 판단"""
        lines = self.raw_text.strip().split("\n")
        print(f"  총 {len(lines)}줄")

        # --- 원문 요약 출력 (Disallow/Allow 규칙만) ---
        print(f"\n  --- 규칙 요약 ---")
        current_ua = "*"
        disallow_rules = []
        allow_rules = []

        for line in lines:
            line_stripped = line.strip()
            if not line_stripped or line_stripped.startswith("#"):
                continue
            if line_stripped.lower().startswith("user-agent:"):
                current_ua = line_stripped.split(":", 1)[1].strip()
            elif line_stripped.lower().startswith("disallow:"):
                path = line_stripped.split(":", 1)[1].strip()
                if path:
                    disallow_rules.append((current_ua, path))
                    print(f"  Disallow [{current_ua}]: {path}")
            elif line_stripped.lower().startswith("allow:"):
                path = line_stripped.split(":", 1)[1].strip()
                if path:
                    allow_rules.append((current_ua, path))
                    print(f"  Allow    [{current_ua}]: {path}")

        if not disallow_rules and not allow_rules:
            print(f"  (규칙 없음 = 전체 허용)")
            self.accessible = True
            return True

        # --- urllib.robotparser로 정밀 판단 ---
        self.parser.parse(lines)

        print(f"\n  --- 경로별 접근 판단 (User-Agent: 일반 브라우저) ---")
        ua = HEADERS["User-Agent"]
        all_allowed = True

        for path in PATHS_TO_CHECK:
            full_url = f"{self.base_url}{path}"
            allowed = self.parser.can_fetch(ua, full_url)
            # 범용 UA(*)로도 체크
            allowed_star = self.parser.can_fetch("*", full_url)
            final = allowed and allowed_star
            status = "✅ 허용" if final else "❌ 차단"
            print(f"  {status}: {path}")
            if not final:
                all_allowed = False

        # 모바일 (m.blog.naver.com은 별도 도메인이므로 여기서는 참고만)
        print(f"\n  참고: m.blog.naver.com 은 별도 도메인 (별도 robots.txt 적용)")

        if all_allowed:
            print(f"\n  ✅ 결론: 필요한 모든 경로가 허용됨")
            self.accessible = True
            return True
        else:
            print(f"\n  ⚠️ 결론: 일부 경로가 차단됨")
            print(f"  → 차단된 경로는 사용하지 않고, 허용된 경로만 사용하세요.")
            print(f"  → 모바일 버전(m.blog.naver.com)이 대안이 될 수 있습니다.")
            # 부분 차단이면 일단 진행 가능 (차단 경로만 회피)
            self.accessible = True
            return True

    def check_path(self, path: str) -> bool:
        """특정 경로가 허용되는지 확인"""
        if not self.raw_text:
            return False
        ua = HEADERS["User-Agent"]
        full_url = f"{self.base_url}{path}"
        return self.parser.can_fetch(ua, full_url) and self.parser.can_fetch("*", full_url)


# ============================================================
# 2. RSS 확인
# ============================================================
def check_rss(blog_id: str) -> bool:
    section("2. RSS 피드 확인")
    url = f"https://rss.blog.naver.com/{blog_id}.xml"
    print(f"  URL: {url}")

    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        print(f"  Status: {resp.status_code}")
        print(f"  Content-Type: {resp.headers.get('Content-Type', 'N/A')}")

        if resp.status_code != 200:
            print(f"  ❌ RSS 피드 접근 실패 (HTTP {resp.status_code})")
            return False

        text = resp.text.strip()
        if not text.startswith("<?xml") and "<rss" not in text[:500]:
            print(f"  ❌ XML/RSS 형식이 아님 (HTML 리다이렉트일 수 있음)")
            if "<html" in text[:500].lower():
                print(f"     HTML 응답 감지 → RSS 비활성 상태")
            return False

        # XML 파싱
        root = ET.fromstring(text)
        channel = root.find(".//channel")
        items = root.findall(".//item")

        blog_title = root.findtext(".//channel/title", "N/A")
        blog_desc = root.findtext(".//channel/description", "N/A")

        print(f"  ✅ RSS 피드 발견!")
        print(f"  블로그 제목: {blog_title}")
        print(f"  설명: {blog_desc}")
        print(f"  글 수: {len(items)}개")

        if items:
            print(f"\n  --- 최근 글 (최대 5개) ---")
            for item in items[:5]:
                t = item.findtext("title", "?")
                link = item.findtext("link", "?")
                pub = item.findtext("pubDate", "?")
                cat = item.findtext("category", "-")
                print(f"  [{cat}] {t}")
                print(f"       날짜: {pub}")
                print(f"       URL:  {link}")

        return True

    except ET.ParseError as e:
        print(f"  ❌ XML 파싱 실패: {e}")
        return False
    except Exception as e:
        print(f"  ❌ 오류: {e}")
        return False


# ============================================================
# 3. 블로그 글 목록 API
# ============================================================
def check_blog_api(blog_id: str) -> bool:
    section("3. 블로그 글 목록 API (PostTitleListAsync)")
    url = (
        f"https://blog.naver.com/PostTitleListAsync.naver"
        f"?blogId={blog_id}&currentPage=1&categoryNo=0&countPerPage=5"
    )
    print(f"  URL: {url}")

    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        print(f"  Status: {resp.status_code}")
        print(f"  응답 크기: {len(resp.text):,} bytes")

        if resp.status_code == 200 and len(resp.text) > 100:
            # logNo 패턴이 있는지 확인
            log_nos = re.findall(r"logNo=(\d+)", resp.text)
            if log_nos:
                print(f"  ✅ 글 목록 API 접근 가능 (logNo {len(log_nos)}개 발견)")
                return True
            else:
                print(f"  ⚠️ 응답은 있으나 글 목록이 비어 있음")
                return False
        elif resp.status_code == 403:
            print(f"  ❌ 403 차단 — 캡차/차단 의심")
            return False
        else:
            print(f"  ❌ 접근 실패")
            return False
    except Exception as e:
        print(f"  ❌ 오류: {e}")
        return False


# ============================================================
# 4. 모바일 접근
# ============================================================
def check_mobile(blog_id: str) -> bool:
    section("4. 모바일 블로그 접근 확인")
    url = f"https://m.blog.naver.com/{blog_id}"
    print(f"  URL: {url}")

    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        print(f"  Status: {resp.status_code}")
        print(f"  응답 크기: {len(resp.text):,} bytes")

        if resp.status_code == 200:
            if "없는 블로그" in resp.text or "존재하지 않는" in resp.text:
                print(f"  ❌ 블로그가 존재하지 않거나 비공개")
                return False
            print(f"  ✅ 모바일 블로그 접근 가능")
            return True
        elif resp.status_code == 403:
            print(f"  ❌ 403 차단")
            return False
        else:
            print(f"  ❌ HTTP {resp.status_code}")
            return False
    except Exception as e:
        print(f"  ❌ 오류: {e}")
        return False


# ============================================================
# 5. 모바일 robots.txt (별도 체크)
# ============================================================
def check_mobile_robots() -> bool:
    section("5. m.blog.naver.com robots.txt")
    url = "https://m.blog.naver.com/robots.txt"
    print(f"  URL: {url}")

    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        print(f"  Status: {resp.status_code}")
        if resp.status_code == 200:
            lines = resp.text.strip().split("\n")
            for line in lines[:20]:
                if line.strip() and not line.strip().startswith("#"):
                    print(f"  {line.strip()}")
            return True
        elif resp.status_code == 404:
            print(f"  robots.txt 없음 (404) = 크롤링 제한 없음")
            return True
        else:
            print(f"  ⚠️ 확인 불가 (HTTP {resp.status_code})")
            return False
    except Exception as e:
        print(f"  ❌ 오류: {e}")
        return False


# ============================================================
# 메인
# ============================================================
def main():
    ap = argparse.ArgumentParser(description="네이버 블로그 사전 점검")
    ap.add_argument("--blog-id", default=DEFAULT_BLOG_ID, help=f"블로그 ID (기본: {DEFAULT_BLOG_ID})")
    args = ap.parse_args()
    blog_id = args.blog_id

    print("╔══════════════════════════════════════════════════════════╗")
    print(f"║  네이버 블로그 사전 점검: {blog_id:<30}║")
    print("╚══════════════════════════════════════════════════════════╝")

    # 1) robots.txt (PC)
    robots = RobotsChecker("https://blog.naver.com")
    robots_ok = robots.fetch_and_parse()

    if not robots_ok:
        print("\n" + "!" * 64)
        print("  ⛔ robots.txt 확인 실패 — 보수적으로 수집을 중단합니다.")
        print("  수동으로 https://blog.naver.com/robots.txt 에 접속하여")
        print("  크롤링 허용 여부를 확인한 뒤 다시 시도하세요.")
        print("!" * 64)
        return 1

    # 2) RSS
    time.sleep(1)
    rss_ok = check_rss(blog_id)

    # 3) 글 목록 API
    time.sleep(1)
    api_ok = check_blog_api(blog_id)

    # 4) 모바일
    time.sleep(1)
    mobile_ok = check_mobile(blog_id)

    # 5) 모바일 robots.txt
    time.sleep(1)
    mobile_robots_ok = check_mobile_robots()

    # ─── 최종 요약 ───
    section("📋 최종 요약")
    print(f"  robots.txt (PC):  {'✅ 확인 완료' if robots_ok else '❌ 실패'}")
    print(f"  robots.txt (모바일): {'✅' if mobile_robots_ok else '❌'}")
    print(f"  RSS 피드:         {'✅' if rss_ok else '❌'}")
    print(f"  글 목록 API:      {'✅' if api_ok else '❌'}")
    print(f"  모바일 접근:      {'✅' if mobile_ok else '❌'}")

    print()
    if rss_ok:
        print("  🟢 추천 모드: RSS")
        print("     python crawler.py --mode rss")
        print()
        print("     RSS가 최신 글만 포함할 수 있으므로,")
        print("     전체 글이 필요하면 --mode auto (RSS+HTML 혼합) 사용")
    elif api_ok and mobile_ok:
        print("  🟡 추천 모드: HTML (모바일 우선)")
        print("     python crawler.py --mode html")
    elif api_ok:
        print("  🟡 추천 모드: HTML (PC API)")
        print("     python crawler.py --mode html")
    elif mobile_ok:
        print("  🟡 추천 모드: HTML (모바일 전용)")
        print("     python crawler.py --mode html")
    else:
        print("  🔴 모든 접근 경로 실패")
        print("     네이버 검색 API(Plan B) 사용을 권장합니다:")
        print("     python crawler.py --mode search-api --naver-client-id YOUR_ID --naver-client-secret YOUR_SECRET")
        print()
        print("     네이버 개발자센터: https://developers.naver.com/apps")

    return 0


if __name__ == "__main__":
    sys.exit(main())