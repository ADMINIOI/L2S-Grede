"""Parser and redirect tests. They do not call TikTok, Instagram, or Facebook."""

from __future__ import annotations

import gzip
import json
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from counts import (  # noqa: E402
    CountUnavailable,
    FACEBOOK_SHARE_URL,
    HttpResult,
    canonical_facebook_url,
    fetch_facebook,
    fetch_instagram,
    fetch_tiktok,
    follower_count_in_text,
    is_facebook_page_url,
    parse_count_token,
    parse_facebook_html,
    parse_instagram_embed,
    parse_instagram_html,
    parse_tiktok_html,
    read_url,
)

TIKTOK_PAGE = """
<script id="__UNIVERSAL_DATA_FOR_REHYDRATION__" type="application/json">%s</script>
"""

INSTAGRAM_PAGE = """
<meta property="og:title" content="AxisBox (&#064;axisbox) &#x2022; Instagram photos and videos" />
<meta property="og:description" content="8 Followers, 3 Following, 6 Posts - See Instagram photos and videos from AxisBox (&#064;axisbox)" />
<meta content="8 Followers, 3 Following, 6 Posts - AxisBox (&#064;axisbox) on Instagram: &quot;&quot;" name="description" />
"""

FACEBOOK_PAGE = """
<meta property="og:title" content="AxisBox" />
<meta property="og:description" content="AxisBox. 1 talking about this. We design and print custom 3D models." />
"""


def instagram_embed(count: int = 8, username: str = "axisbox") -> str:
    payload = {"context": {"username": username, "full_name": "AxisBox", "followers_count": count, "posts_count": 6}}
    return 'contextJSON":' + json.dumps(json.dumps(payload))


def tiktok_payload(count: int = 3, mirror: str = "3", unique_id: str = "axisbox.am") -> str:
    payload = {
        "__DEFAULT_SCOPE__": {
            "webapp.user-detail": {
                "statusCode": 0,
                "userInfo": {
                    "user": {"uniqueId": unique_id, "nickname": "AxisBox"},
                    "stats": {"followerCount": count},
                    "statsV2": {"followerCount": mirror},
                },
            }
        }
    }
    return json.dumps(payload)


class ParseTests(unittest.TestCase):
    def test_exact_count_tokens(self) -> None:
        self.assertEqual(parse_count_token("8"), 8)
        self.assertEqual(parse_count_token("1,234"), 1234)
        self.assertEqual(parse_count_token("1.234.567"), 1234567)
        self.assertEqual(parse_count_token("1\u00a0234"), 1234)
        self.assertEqual(parse_count_token("1.2K"), 1200)
        self.assertEqual(parse_count_token("1,2 тыс."), 1200)
        self.assertEqual(parse_count_token("2M"), 2_000_000)

    def test_rejects_ambiguous_tokens(self) -> None:
        with self.assertRaises(ValueError):
            parse_count_token("1.2")
        with self.assertRaises(ValueError):
            parse_count_token("about")

    def test_tiktok_count_comes_from_stats(self) -> None:
        handle, count = parse_tiktok_html(TIKTOK_PAGE % tiktok_payload())
        self.assertEqual(handle, "@axisbox.am")
        self.assertEqual(count, 3)

    def test_tiktok_mismatch_is_an_error(self) -> None:
        page = TIKTOK_PAGE % tiktok_payload(count=3, mirror="4")
        with self.assertRaises(CountUnavailable):
            parse_tiktok_html(page)

    def test_instagram_embed_count_is_for_this_username(self) -> None:
        handle, count = parse_instagram_embed(instagram_embed())
        self.assertEqual(handle, "@axisbox")
        self.assertEqual(count, 8)

    def test_instagram_embed_ignores_a_different_username(self) -> None:
        with self.assertRaises(CountUnavailable):
            parse_instagram_embed(instagram_embed(username="someone_else"))

    def test_instagram_public_preview_count(self) -> None:
        handle, count = parse_instagram_html(INSTAGRAM_PAGE)
        self.assertEqual(handle, "@axisbox")
        self.assertEqual(count, 8)

    def test_instagram_ignores_following_and_posts(self) -> None:
        self.assertEqual(follower_count_in_text("8 Followers, 3 Following, 6 Posts"), 8)

    def test_russian_follower_word(self) -> None:
        self.assertEqual(follower_count_in_text("8 подписчиков, 3 подписки"), 8)

    def test_facebook_talking_about_is_not_a_follower_count(self) -> None:
        title, count = parse_facebook_html(FACEBOOK_PAGE)
        self.assertEqual(title, "AxisBox")
        self.assertIsNone(count)
        text = "AxisBox. 1 talking about this. We design and print custom 3D models. Prototypes, parts, gifts — fast, precise, creative."
        self.assertIsNone(follower_count_in_text(text))

    def test_facebook_likes_are_not_followers(self) -> None:
        html = '<meta property="og:description" content="AxisBox. 4 likes. 1 talking about this." />'
        _title, count = parse_facebook_html(html)
        self.assertIsNone(count)

    def test_facebook_social_context_is_this_page_only(self) -> None:
        page = "https://www.facebook.com/AxisBox/"
        html = (
            '"profile_social_context":{"content":['
            '{"text":{"text":"0 followers"},"uri":"https:\\/\\/www.facebook.com\\/AxisBox\\/followers\\/"},'
            '{"text":{"text":"0 following"},"uri":"https:\\/\\/www.facebook.com\\/AxisBox\\/following\\/"},'
            '{"text":{"text":"9 followers"},"uri":"https:\\/\\/www.facebook.com\\/Other\\/followers\\/"}'
            "]}"
        )
        title, count = parse_facebook_html(html, page)
        self.assertIsNone(title)
        self.assertEqual(count, 0)

    def test_facebook_followers_are_read(self) -> None:
        html = '<meta property="og:title" content="AxisBox" /><meta property="og:description" content="AxisBox. 12 followers. 1 talking about this." />'
        title, count = parse_facebook_html(html)
        self.assertEqual(title, "AxisBox")
        self.assertEqual(count, 12)

    def test_disagreeing_descriptions_are_rejected(self) -> None:
        text = "8 Followers\n9 Followers"
        with self.assertRaises(CountUnavailable):
            follower_count_in_text(text)

    def test_share_url_is_not_a_page(self) -> None:
        self.assertFalse(is_facebook_page_url(FACEBOOK_SHARE_URL))
        self.assertTrue(is_facebook_page_url("https://www.facebook.com/AxisBox?rdid=abc"))
        self.assertEqual(
            canonical_facebook_url("https://www.facebook.com/AxisBox?rdid=abc"),
            "https://www.facebook.com/AxisBox/",
        )


class FetchTests(unittest.TestCase):
    def test_offsite_redirect_is_not_a_page(self) -> None:
        def reader(url, headers, *, follow, timeout):
            return HttpResult(302, {"Location": "https://evil.example/AxisBox"}, "", url)

        result = fetch_facebook(reader)
        self.assertIsNone(result.count)
        self.assertEqual(result.error, "ссылка не открылась")

    def test_requests_do_not_send_cookies(self) -> None:
        seen: list[dict[str, str]] = []

        def reader(url, headers, *, follow, timeout):
            seen.append(dict(headers))
            return HttpResult(500, {}, "", url)

        fetch_tiktok(reader)
        fetch_instagram(reader)
        fetch_facebook(reader)
        self.assertTrue(seen)
        for headers in seen:
            self.assertNotIn("cookie", {key.lower() for key in headers})

    def test_instagram_stops_when_the_embed_has_a_count(self) -> None:
        calls: list[str] = []

        def reader(url, headers, *, follow, timeout):
            calls.append(url)
            return HttpResult(200, {}, instagram_embed(), url)

        result = fetch_instagram(reader)
        self.assertEqual(result.count, 8)
        self.assertEqual(result.handle, "@axisbox")
        self.assertIsNone(result.error)
        self.assertEqual(calls, ["https://www.instagram.com/axisbox/embed/"])

    def test_instagram_uses_preview_markup_after_a_login_wall(self) -> None:
        def reader(url, headers, *, follow, timeout):
            if "externalhit" not in headers["User-Agent"]:
                return HttpResult(200, {}, "<title>Login</title>", url + "accounts/login/")
            return HttpResult(200, {}, INSTAGRAM_PAGE, url)

        result = fetch_instagram(reader)
        self.assertEqual(result.count, 8)
        self.assertEqual(result.handle, "@axisbox")
        self.assertIsNone(result.error)

    def test_tiktok_fetch_uses_the_json_count(self) -> None:
        def reader(url, headers, *, follow, timeout):
            return HttpResult(200, {}, TIKTOK_PAGE % tiktok_payload(count=3), url)

        result = fetch_tiktok(reader)
        self.assertEqual(result.count, 3)
        self.assertEqual(result.handle, "@axisbox.am")
        self.assertIsNone(result.error)


class RedirectServerTests(unittest.TestCase):
    def test_share_redirect_resolves_and_does_not_invent_a_count(self) -> None:
        pages = {"description": "AxisBox. 1 talking about this."}

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                path = self.path.split("?", 1)[0]
                if path.startswith("/share"):
                    self.send_response(302)
                    self.send_header("Location", "/AxisBox?rdid=1&share_url=1")
                    self.end_headers()
                    return
                body = (
                    '<meta property="og:title" content="AxisBox" />'
                    f'<meta property="og:description" content="{pages["description"]}" />'
                ).encode()
                data = gzip.compress(body)
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Encoding", "gzip")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, fmt: str, *args) -> None:
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            host, port = server.server_address
            share = f"http://{host}:{port}/share/1BtnCAHH58/"
            hidden = fetch_facebook(read_url, share_url=share)
            self.assertIsNone(hidden.count)
            self.assertEqual(hidden.handle, "AxisBox")
            self.assertEqual(hidden.page_url, f"http://{host}:{port}/AxisBox/")
            self.assertEqual(hidden.error, "публичная страница без счётчика")

            pages["description"] = "AxisBox. 12 followers. 1 talking about this."
            shown = fetch_facebook(read_url, share_url=share)
            self.assertEqual(shown.count, 12)
            self.assertIsNone(shown.error)
            self.assertEqual(shown.handle, "AxisBox")
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
