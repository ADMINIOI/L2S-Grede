"""Public follower counts for the AxisBox profiles.

Counts come only from public profile HTML. There is no API token, login, or
cookie jar. A missing public number becomes an error string, never a guess.

TikTok includes followerCount in the page's rehydration JSON for a normal
browser request. Instagram's profile URL is often a login wall; its public
profile embed states followers_count for that username, and the profile URL's
link-preview description ("N Followers") is used when the embed has no count.
Facebook's share link is followed to the canonical page. A follower count is
read only from that page's own "N followers" text (the profile social context
or the public description). "talking about this", "following", and page likes
are not follower counts.
"""

from __future__ import annotations

import gzip
import json
import re
import socket
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from html import unescape
from typing import Callable, Mapping
from urllib.parse import parse_qs, urljoin, urlparse, urlunparse

TIKTOK_URL = "https://www.tiktok.com/@axisbox.am"
INSTAGRAM_URL = "https://www.instagram.com/axisbox/"
INSTAGRAM_EMBED_URL = "https://www.instagram.com/axisbox/embed/"
INSTAGRAM_USERNAME = "axisbox"
FACEBOOK_SHARE_URL = "https://www.facebook.com/share/1BtnCAHH58/"

REFRESH_SECONDS = 180
MAX_BYTES = 3_000_000
TIMEOUT = 20.0

BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)
# Public link-preview client. Instagram serves the follower count in this
# document and withholds it from a normal browser session.
PREVIEW_UA = "facebookexternalhit/1.1 (+http://www.facebook.com/externalhit_uatext.php)"

_COUNT_TOKEN = r"(\d[\d\s\u00a0.,]*(?:\s*[KkMmBb]|\s*(?:тыс|млн)\.?)?)"
_FOLLOWER_PATTERNS = (
    re.compile(_COUNT_TOKEN + r"\s+Followers?\b", re.IGNORECASE),
    re.compile(_COUNT_TOKEN + r"\s+people follow this\b", re.IGNORECASE),
    re.compile(_COUNT_TOKEN + r"\s+подписчик(?:а|ов|и)?\b", re.IGNORECASE),
)
_META_TAG = re.compile(r"<meta\b[^>]*>", re.IGNORECASE)
_ATTR = re.compile(r"""\b([A-Za-z_:][-A-Za-z0-9_:.]*)\s*=\s*(['"])(.*?)\2""", re.IGNORECASE | re.DOTALL)
_TIKTOK_JSON = re.compile(
    r'<script id="__UNIVERSAL_DATA_FOR_REHYDRATION__" type="application/json">(.*?)</script>',
    re.DOTALL,
)
_IG_HANDLE = re.compile(r"@([A-Za-z0-9._]+)")

Reader = Callable[..., "HttpResult"]


class CountUnavailable(Exception):
    def __init__(self, message: str, handle: str | None = None, page_url: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.handle = handle
        self.page_url = page_url


@dataclass(frozen=True)
class HttpResult:
    status: int
    headers: Mapping[str, str]
    body: str
    url: str

    def header(self, name: str) -> str | None:
        for key, value in self.headers.items():
            if key.lower() == name.lower():
                return value
        return None


@dataclass(frozen=True)
class ProfileCount:
    platform: str
    handle: str
    count: int | None
    error: str | None
    page_url: str

    def to_dict(self) -> dict[str, object]:
        return {
            "platform": self.platform,
            "handle": self.handle,
            "count": self.count,
            "error": self.error,
            "page_url": self.page_url,
        }


class _NoErrorProcessor(urllib.request.HTTPErrorProcessor):
    """Return 3xx/4xx responses so callers can read a redirect Location."""

    def http_response(self, request, response):  # type: ignore[no-untyped-def]
        return response

    https_response = http_response


def browser_headers() -> dict[str, str]:
    return {
        "User-Agent": BROWSER_UA,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9,ru;q=0.8",
        "Accept-Encoding": "gzip, identity",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Upgrade-Insecure-Requests": "1",
    }


def preview_headers() -> dict[str, str]:
    return {
        "User-Agent": PREVIEW_UA,
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, identity",
    }


def parse_count_token(token: str) -> int:
    """Parse a public count token. Raises ValueError when it is not exact."""
    raw = token.replace("\u00a0", " ").strip().lower()
    compact = raw.replace(" ", "").rstrip(".")
    multiplier = 1
    for suffix, factor in (("млн", 1_000_000), ("тыс", 1_000), ("k", 1_000), ("m", 1_000_000), ("b", 1_000_000_000)):
        if compact.endswith(suffix):
            multiplier = factor
            compact = compact[: -len(suffix)].rstrip(".")
            break
    if multiplier != 1:
        if re.fullmatch(r"\d{1,3}(?:[.,]\d{3})+", compact):
            base: float = int(compact.replace(",", "").replace(".", ""))
        elif re.fullmatch(r"\d+(?:[.,]\d{1,2})?", compact):
            base = float(compact.replace(",", "."))
        else:
            raise ValueError(token)
        value = int(round(base * multiplier))
    elif re.fullmatch(r"\d{1,3}(?:[ .,\u00a0]\d{3})+", raw):
        value = int(re.sub(r"[ .,\u00a0]", "", raw))
    elif re.fullmatch(r"\d+", compact):
        value = int(compact)
    else:
        raise ValueError(token)
    if value < 0:
        raise ValueError(token)
    return value


def follower_count_in_text(text: str) -> int | None:
    """Return a follower count stated in text, or None when none is stated."""
    found: list[int] = []
    for pattern in _FOLLOWER_PATTERNS:
        for match in pattern.finditer(text):
            try:
                found.append(parse_count_token(match.group(1)))
            except ValueError:
                continue
    if not found:
        return None
    if any(item != found[0] for item in found):
        raise CountUnavailable("счётчик не совпал")
    return found[0]


def _meta_map(tag: str) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for match in _ATTR.finditer(tag):
        attrs[match.group(1).lower()] = unescape(match.group(3))
    return attrs


def description_texts(html: str) -> list[str]:
    texts: list[str] = []
    for tag in _META_TAG.findall(html):
        attrs = _meta_map(tag)
        key = (attrs.get("property") or attrs.get("name") or "").lower()
        if key in {"og:description", "description", "twitter:description"} and attrs.get("content"):
            texts.append(attrs["content"])
    return texts


def og_title(html: str) -> str | None:
    for tag in _META_TAG.findall(html):
        attrs = _meta_map(tag)
        if (attrs.get("property") or "").lower() == "og:title" and attrs.get("content"):
            title = attrs["content"].strip()
            if title:
                return title
    return None


def parse_tiktok_html(html: str) -> tuple[str, int]:
    match = _TIKTOK_JSON.search(html)
    if not match:
        raise CountUnavailable("счётчик не найден")
    try:
        payload = json.loads(match.group(1))
        detail = payload["__DEFAULT_SCOPE__"]["webapp.user-detail"]
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise CountUnavailable("счётчик не найден") from exc
    if detail.get("statusCode") not in (0, None):
        raise CountUnavailable("счётчик не найден")
    try:
        user = detail["userInfo"]["user"]
        stats = detail["userInfo"]["stats"]
        count = stats["followerCount"]
        unique_id = user["uniqueId"]
    except (KeyError, TypeError) as exc:
        raise CountUnavailable("счётчик не найден") from exc
    if isinstance(count, str) and count.isdigit():
        count = int(count)
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise CountUnavailable("счётчик не найден")
    stats_v2 = detail["userInfo"].get("statsV2") or {}
    mirror = stats_v2.get("followerCount")
    if mirror is not None and str(mirror) != str(count):
        raise CountUnavailable("счётчик не совпал")
    if not isinstance(unique_id, str) or not unique_id:
        raise CountUnavailable("счётчик не найден")
    return f"@{unique_id}", count


def _as_count(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def parse_instagram_embed(html: str, username: str = INSTAGRAM_USERNAME) -> tuple[str, int]:
    """Read followers_count from the public profile embed for this username."""
    found: list[tuple[str, int]] = []
    start = 0
    while True:
        marker = html.find("contextJSON", start)
        if marker < 0:
            break
        colon = html.find(":", marker)
        quote = html.find('"', colon if colon >= 0 else marker)
        if quote < 0:
            break
        try:
            encoded, _end = json.JSONDecoder().raw_decode(html[quote:])
            payload = json.loads(encoded) if isinstance(encoded, str) else encoded
        except (json.JSONDecodeError, TypeError, ValueError):
            start = marker + len("contextJSON")
            continue
        context = payload.get("context") if isinstance(payload, dict) else None
        if isinstance(context, dict):
            found_name = context.get("username")
            count = _as_count(context.get("followers_count"))
            if isinstance(found_name, str) and found_name.lower() == username.lower() and count is not None:
                found.append((found_name, count))
        start = marker + len("contextJSON")
    if not found:
        raise CountUnavailable("счётчик скрыт без входа")
    if any(item[1] != found[0][1] for item in found):
        raise CountUnavailable("счётчик не совпал")
    return f"@{found[0][0]}", found[0][1]


def parse_instagram_html(html: str) -> tuple[str | None, int]:
    texts = description_texts(html)
    if not texts:
        raise CountUnavailable("счётчик скрыт без входа")
    count = follower_count_in_text("\n".join(texts))
    if count is None:
        raise CountUnavailable("счётчик скрыт без входа")
    handle = None
    title = og_title(html) or ""
    found = _IG_HANDLE.search(title)
    if found:
        handle = f"@{found.group(1)}"
    return handle, count


def _page_followers_uri(uri: str, page_url: str) -> bool:
    uri_parts = [part.lower() for part in urlparse(uri).path.split("/") if part]
    page_parts = [part.lower() for part in urlparse(page_url).path.split("/") if part]
    return bool(page_parts) and uri_parts[-1:] == ["followers"] and uri_parts[:-1] == page_parts


def facebook_social_follower_count(html: str, page_url: str) -> int | None:
    """Read the count from this page's own followers link, not a nearby page."""
    pattern = re.compile(r'"text":("(?:\\.|[^"\\])*")\s*}\s*,\s*"uri":("(?:\\.|[^"\\])*")')
    found: list[int] = []
    for match in pattern.finditer(html):
        try:
            text = json.loads(match.group(1))
            uri = json.loads(match.group(2))
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(text, str) or not isinstance(uri, str):
            continue
        if not _page_followers_uri(uri, page_url):
            continue
        count = follower_count_in_text(text)
        if count is not None:
            found.append(count)
    if not found:
        return None
    if any(item != found[0] for item in found):
        raise CountUnavailable("счётчик не совпал")
    return found[0]


def parse_facebook_html(html: str, page_url: str | None = None) -> tuple[str | None, int | None]:
    title = og_title(html)
    if title and title.lower() in {"facebook", "log in", "login", "error"}:
        title = None
    texts = description_texts(html)
    described = follower_count_in_text("\n".join(texts)) if texts else None
    linked = facebook_social_follower_count(html, page_url) if page_url else None
    if described is not None and linked is not None and described != linked:
        raise CountUnavailable("счётчик не совпал")
    if linked is not None:
        return title, linked
    return title, described


def _blocked_facebook_path(parts: list[str]) -> bool:
    if not parts:
        return True
    head = parts[0].lower()
    return head in {
        "login",
        "login.php",
        "share",
        "checkpoint",
        "recover",
        "home.php",
        "privacy",
        "policies",
        "settings",
        "watch",
        "marketplace",
        "gaming",
        "reel",
        "reels",
        "stories",
        "dialog",
        "sharer",
    }


def _is_facebook_host(host: str) -> bool:
    name = host.lower().split(":")[0]
    return name == "facebook.com" or name.endswith(".facebook.com")


def is_facebook_page_url(url: str, origin: str | None = None) -> bool:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    origin_host = urlparse(origin).netloc.lower() if origin else ""
    if not (_is_facebook_host(host) or (origin_host and host == origin_host)):
        return False
    parts = [part for part in parsed.path.split("/") if part]
    if _blocked_facebook_path(parts):
        return False
    head = parts[0].lower()
    if head == "profile.php":
        return "id" in parse_qs(parsed.query)
    if head in {"pages", "people", "pg"}:
        return len(parts) >= 2
    return len(parts) == 1


def canonical_facebook_url(url: str) -> str:
    parsed = urlparse(url)
    parts = [part for part in parsed.path.split("/") if part]
    query = ""
    if parts and parts[0].lower() == "profile.php":
        profile_id = parse_qs(parsed.query).get("id", [""])[0]
        path = "/profile.php"
        query = f"id={profile_id}"
    else:
        path = "/" + "/".join(parts) + "/"
    return urlunparse((parsed.scheme or "https", parsed.netloc, path, "", query, ""))


def resolve_facebook_page(share_url: str, reader: Reader) -> str:
    url = share_url
    for _ in range(6):
        result = reader(url, browser_headers(), follow=False, timeout=TIMEOUT)
        location = result.header("Location") if result.status in {301, 302, 303, 307, 308} else None
        if not location:
            if is_facebook_page_url(result.url, share_url):
                return canonical_facebook_url(result.url)
            raise CountUnavailable("ссылка не открылась", page_url=share_url)
        target = urljoin(url, location.strip())
        if is_facebook_page_url(target, share_url):
            return canonical_facebook_url(target)
        if urlparse(target).path.lower().startswith("/share"):
            url = target
            continue
        raise CountUnavailable("ссылка не открылась", page_url=share_url)
    raise CountUnavailable("ссылка не открылась", page_url=share_url)


def _decode_body(raw: bytes, encoding: str | None) -> str:
    chosen = (encoding or "").lower()
    if "gzip" in chosen:
        try:
            raw = gzip.decompress(raw)
        except OSError as exc:
            raise CountUnavailable("счётчик не найден") from exc
    elif chosen and "identity" not in chosen and chosen != "":
        raise CountUnavailable("счётчик не найден")
    return raw.decode("utf-8", "replace")


def _header_pairs(headers: Mapping[str, str]) -> dict[str, str]:
    if hasattr(headers, "items"):
        return {str(key): str(value) for key, value in headers.items()}
    return {}


def read_url(url: str, headers: Mapping[str, str], *, follow: bool, timeout: float = TIMEOUT) -> HttpResult:
    request = urllib.request.Request(url, headers=dict(headers), method="GET")
    opener = urllib.request.build_opener() if follow else urllib.request.build_opener(_NoErrorProcessor)
    try:
        with opener.open(request, timeout=timeout) as response:
            raw = response.read(MAX_BYTES + 1)
            if len(raw) > MAX_BYTES:
                raise CountUnavailable("счётчик не найден")
            body = _decode_body(raw, response.headers.get("Content-Encoding"))
            return HttpResult(response.status, _header_pairs(response.headers), body, response.geturl())
    except urllib.error.HTTPError as exc:
        raw = exc.read(MAX_BYTES + 1)
        body = _decode_body(raw, exc.headers.get("Content-Encoding")) if raw else ""
        return HttpResult(exc.code, _header_pairs(exc.headers), body, exc.geturl())
    except (urllib.error.URLError, socket.timeout, TimeoutError) as exc:
        raise CountUnavailable("нет ответа") from exc


def _result(platform: str, handle: str, page_url: str, count: int | None, error: str | None) -> ProfileCount:
    if count is not None and error is not None:
        raise ValueError("a profile cannot have both a count and an error")
    if count is None and not error:
        raise ValueError("a profile needs a count or an error")
    return ProfileCount(platform=platform, handle=handle, count=count, error=error, page_url=page_url)


def _from_unavailable(platform: str, fallback_handle: str, fallback_url: str, exc: CountUnavailable) -> ProfileCount:
    return _result(
        platform,
        exc.handle or fallback_handle,
        exc.page_url or fallback_url,
        None,
        exc.message,
    )


def fetch_tiktok(reader: Reader = read_url) -> ProfileCount:
    try:
        result = reader(TIKTOK_URL, browser_headers(), follow=True, timeout=TIMEOUT)
        if result.status != 200:
            raise CountUnavailable("нет ответа")
        handle, count = parse_tiktok_html(result.body)
        return _result("TikTok", handle, TIKTOK_URL, count, None)
    except CountUnavailable as exc:
        return _from_unavailable("TikTok", "@axisbox.am", TIKTOK_URL, exc)


def _instagram_page(reader: Reader, url: str, headers: Mapping[str, str]) -> HttpResult | None:
    try:
        result = reader(url, headers, follow=True, timeout=TIMEOUT)
    except CountUnavailable:
        return None
    if result.status != 200 or "/accounts/login" in result.url:
        return None
    return result


def fetch_instagram(reader: Reader = read_url) -> ProfileCount:
    last_error = CountUnavailable("счётчик скрыт без входа")
    embed = _instagram_page(reader, INSTAGRAM_EMBED_URL, browser_headers())
    if embed is not None:
        try:
            handle, count = parse_instagram_embed(embed.body)
            return _result("Instagram", handle, INSTAGRAM_URL, count, None)
        except CountUnavailable as exc:
            last_error = exc
    for headers in (browser_headers(), preview_headers()):
        page = _instagram_page(reader, INSTAGRAM_URL, headers)
        if page is None:
            continue
        try:
            handle, count = parse_instagram_html(page.body)
        except CountUnavailable as exc:
            last_error = exc
            continue
        return _result("Instagram", handle or f"@{INSTAGRAM_USERNAME}", INSTAGRAM_URL, count, None)
    return _from_unavailable("Instagram", f"@{INSTAGRAM_USERNAME}", INSTAGRAM_URL, last_error)


def _facebook_handle(page_url: str, title: str | None) -> str:
    if title:
        return title
    parts = [part for part in urlparse(page_url).path.split("/") if part]
    if not parts:
        return "Facebook"
    if parts[0].lower() == "profile.php":
        return "Facebook"
    return parts[-1]


def _facebook_public_document(reader: Reader, page_url: str) -> HttpResult:
    result: HttpResult | None = None
    for attempt in range(2):
        result = reader(page_url, preview_headers(), follow=False, timeout=TIMEOUT)
        location = urljoin(page_url, result.header("Location") or "")
        login_redirect = result.status in {301, 302, 303, 307, 308} and "/login" in urlparse(location).path
        if result.status == 200 or attempt == 1 or not login_redirect:
            return result
        time.sleep(0.4)
    assert result is not None
    return result


def fetch_facebook(reader: Reader = read_url, share_url: str = FACEBOOK_SHARE_URL) -> ProfileCount:
    try:
        page_url = resolve_facebook_page(share_url, reader)
    except CountUnavailable as exc:
        return _from_unavailable("Facebook", "Facebook", share_url, exc)
    handle = _facebook_handle(page_url, None)
    try:
        # Do not follow redirects. A normal browser redirect lands on login,
        # while the first public response already contains the page markup.
        # Facebook sometimes answers the first request with that redirect and
        # the next one with the page, so a login redirect is tried once more.
        result = _facebook_public_document(reader, page_url)
    except CountUnavailable as exc:
        exc.handle = exc.handle or handle
        exc.page_url = exc.page_url or page_url
        return _from_unavailable("Facebook", handle, page_url, exc)
    if result.status in {301, 302, 303, 307, 308}:
        location = urljoin(page_url, result.header("Location") or "")
        if "/login" in urlparse(location).path:
            return _result("Facebook", handle, page_url, None, "страница требует вход")
        return _result("Facebook", handle, page_url, None, "нет ответа")
    if result.status != 200:
        return _result("Facebook", handle, page_url, None, "нет ответа")
    try:
        title, count = parse_facebook_html(result.body, page_url)
    except CountUnavailable as exc:
        exc.handle = exc.handle or handle
        exc.page_url = page_url
        return _from_unavailable("Facebook", handle, page_url, exc)
    handle = _facebook_handle(page_url, title)
    if count is None:
        return _result("Facebook", handle, page_url, None, "публичная страница без счётчика")
    return _result("Facebook", handle, page_url, count, None)


def fetch_all(reader: Reader = read_url) -> list[ProfileCount]:
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = (
            pool.submit(fetch_tiktok, reader),
            pool.submit(fetch_instagram, reader),
            pool.submit(fetch_facebook, reader),
        )
        return [future.result() for future in futures]
