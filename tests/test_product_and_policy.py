import unittest
from unittest.mock import patch

import requests

from shopee_short.audit import AuditLog
from shopee_short.errors import InvalidProductLink, SourceBlocked
from shopee_short.network import _looks_like_challenge, assert_safe_request_url, safe_request
from shopee_short.product import parse_product_link


class ProductAndPolicyTests(unittest.TestCase):
    def test_slug_is_parsed_locally_and_item_ids_are_extracted(self):
        p = parse_product_link(
            "https://shopee.com.br/MAGCUBIC-Projector-HY300-Pro-290-ANSI-i.1633657826.22494542720"
        )
        self.assertEqual(p.name, "MAGCUBIC Projector HY300 Pro 290 ANSI")
        self.assertEqual(p.shop_id, "1633657826")
        self.assertEqual(p.item_id, "22494542720")

    def test_non_shopee_input_fails(self):
        with self.assertRaises(InvalidProductLink):
            parse_product_link("https://example.org/product")

    def test_shopee_and_susercontent_hosts_are_never_fetchable(self):
        for url in (
            "https://shopee.com.br/item",
            "https://cf.shopee.com.br/file/abc",
            "https://down-br.img.susercontent.com/file/abc",
        ):
            with self.subTest(url=url), self.assertRaises(SourceBlocked):
                assert_safe_request_url(url, operation="test")

    def test_shp_ee_resolver_stops_before_shopee_host(self):
        response = requests.Response()
        response.status_code = 302
        response.headers["Location"] = (
            "https://shopee.com.br/Produto-Teste-i.123.456"
        )
        response._content = b""
        audit = AuditLog()
        with patch("requests.Session.request", return_value=response) as request:
            p = parse_product_link("https://shp.ee/abc123", audit=audit)
        self.assertEqual(p.item_id, "456")
        self.assertEqual(audit.request_hosts, ["shp.ee"])
        self.assertEqual(audit.shopee_hosts, [])
        self.assertEqual(request.call_count, 1)
        self.assertEqual(request.call_args.kwargs.get("allow_redirects"), False)

    def test_external_redirect_to_shopee_is_blocked_without_following(self):
        response = requests.Response()
        response.status_code = 302
        response.headers["Location"] = "https://cf.shopee.com.br/file/photo"
        response._content = b""
        session = requests.Session()
        with patch.object(session, "request", return_value=response) as request:
            with self.assertRaises(SourceBlocked):
                safe_request(session, "GET", "https://images.example.org/photo.jpg", operation="test")
        self.assertEqual(request.call_count, 1)
        self.assertFalse(request.call_args.kwargs.get("allow_redirects", True))

    def test_http_403_fails_without_retry(self):
        response = requests.Response()
        response.status_code = 403
        response._content = b"Forbidden"
        session = requests.Session()
        with patch.object(session, "request", return_value=response) as request:
            with self.assertRaises(SourceBlocked):
                safe_request(session, "GET", "https://images.example.org/photo.jpg", operation="test")
        self.assertEqual(request.call_count, 1)

    def test_optional_hcaptcha_script_is_not_mistaken_for_an_active_challenge(self):
        response = requests.Response()
        response.status_code = 200
        response.headers["content-type"] = "text/html"
        response._content = b'<script src="/storefront-forms-hcaptcha.js"></script>'
        self.assertFalse(_looks_like_challenge(response))

    def test_active_human_verification_interstitial_is_detected(self):
        response = requests.Response()
        response.status_code = 200
        response.headers["content-type"] = "text/html"
        response._content = b"Please verify you are human to continue."
        self.assertTrue(_looks_like_challenge(response))


if __name__ == "__main__":
    unittest.main()
